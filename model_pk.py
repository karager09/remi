import tensorflow as tf
import numpy as np
import miditoolkit
import modules
import pickle
import utils_pk_chorales as utils_pk
import time

TRANSPONE = [-3, -2, -1, 0, 1, 2, 3]

class PopMusicTransformer(object):
    ########################################
    # initialize
    ########################################
    def __init__(self, checkpoint, model_name="", is_training=False):
        # load dictionary
        tf.compat.v1.reset_default_graph()
        # self.dictionary_path = '{}/dictionary.pkl'.format(checkpoint)
        # self.event2word, self.word2event = pickle.load(open(self.dictionary_path, 'rb'))
        self.event2word, self.word2event = utils_pk.make_map()
        # model settings
        self.group_size = 5
        self.x_len = 256
        self.mem_len = 256
        # self.x_len = 512
        # self.mem_len = 512
        self.n_layer = 12
        self.d_embed = 512
        self.d_model = 512
        self.dropout = 0.1
        self.n_head = 8
        self.d_head = self.d_model // self.n_head
        self.d_ff = 2048
        self.n_token = len(self.event2word)
        self.learning_rate = 0.0002
        # load model
        self.is_training = is_training
        if self.is_training:
            self.batch_size = 2
        else:
            self.batch_size = 1
        self.checkpoint_path = '{}/{}'.format(checkpoint, model_name)
        self.load_model(model_name, is_training)

    ########################################
    # load model
    ########################################
    def load_model(self, model_name, is_training):
        # placeholders
        self.x = tf.compat.v1.placeholder(tf.int32, shape=[self.batch_size, None])
        self.y = tf.compat.v1.placeholder(tf.int32, shape=[self.batch_size, None])
        self.mems_i = [tf.compat.v1.placeholder(tf.float32, [self.mem_len, self.batch_size, self.d_model]) for _ in range(self.n_layer)]
        # model
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        initializer = tf.compat.v1.initializers.random_normal(stddev=0.02, seed=None)
        proj_initializer = tf.compat.v1.initializers.random_normal(stddev=0.01, seed=None)
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
            xx = tf.transpose(self.x, [1, 0])
            yy = tf.transpose(self.y, [1, 0])
            loss, self.logits, self.new_mem = modules.transformer(
                dec_inp=xx,
                target=yy,
                mems=self.mems_i,
                n_token=self.n_token,
                n_layer=self.n_layer,
                d_model=self.d_model,
                d_embed=self.d_embed,
                n_head=self.n_head,
                d_head=self.d_head,
                d_inner=self.d_ff,
                dropout=self.dropout,
                dropatt=self.dropout,
                initializer=initializer,
                proj_initializer=proj_initializer,
                is_training=self.is_training,
                mem_len=self.mem_len,
                cutoffs=[],
                div_val=-1,
                tie_projs=[],
                same_length=False,
                clamp_len=-1,
                input_perms=None,
                target_perms=None,
                head_target=None,
                untie_r=False,
                proj_same_dim=True)
        self.avg_loss = tf.reduce_mean(loss)
        # vars
        all_vars = tf.compat.v1.trainable_variables()
        grads = tf.gradients(self.avg_loss, all_vars)
        grads_and_vars = list(zip(grads, all_vars))
        all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.compat.v1.trainable_variables()])
        # optimizer
        decay_lr = tf.compat.v1.train.cosine_decay(
            self.learning_rate,
            global_step=self.global_step,
            decay_steps=400000,
            alpha=0.004)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=decay_lr)
        self.train_op = optimizer.apply_gradients(grads_and_vars, self.global_step)
        # saver
        self.saver = tf.compat.v1.train.Saver(max_to_keep=5)
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        if model_name != "":
            self.saver.restore(self.sess, self.checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())

    ########################################
    # temperature sampling
    ########################################
    def temperature_sampling(self, logits, temperature, topk):
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        if topk == 1:
            prediction = np.argmax(probs)
        else:
            sorted_index = np.argsort(probs)[::-1]
            candi_index = sorted_index[:topk]
            candi_probs = [probs[i] for i in candi_index]
            # normalize probs
            candi_probs /= sum(candi_probs)
            # choose by predicted probs
            prediction = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return prediction

    ########################################
    # extract events for prompt continuation
    ########################################
    def extract_events(self, input_path):
        midi_obj = miditoolkit.midi.parser.MidiFile(input_path)
        ticks_per_beat = midi_obj.ticks_per_beat
        time_signature_changes = midi_obj.time_signature_changes

        note_items, tempo_items = utils_pk.read_items(midi_obj)
        note_items = utils_pk.quantize_items(note_items)
        max_time = note_items[-1].end
        if 'chord' in self.checkpoint_path:
            chord_items = utils_pk.extract_chords(note_items)
            items = chord_items + tempo_items + note_items
        else:
            items = tempo_items + note_items
        groups = utils_pk.group_items(items, max_time, ticks_per_beat, time_signature_changes=time_signature_changes)
        events = utils_pk.item2event(groups)
        return events

    ########################################
    # generate
    ########################################
    def generate(self, n_target_bar, temperature, topk, output_path, prompt=None, generate_voice=None):
        # if prompt, load it. Or, random start
        how_many_events = 50
        events_without_voices = None
        if generate_voice is not None:
            meter = 0
            current_position = 1
            last_generated_position = [1, 1]
            events = self.extract_events(prompt)
            events_without_voices = []
            for i in range(len(events)-3):
                if events[i].name == 'Bar':
                    events_without_voices.append(events[i])
                elif events[i].name == 'Position Bar' and \
                    events[i+1].name == 'Position Quarter' and \
                    events[i+2].name == 'Note On' and \
                    events[i+3].name == 'Note Duration' and \
                    events[i+4].name == 'Note Voice':
                    if meter < int(events[i].value):
                        meter = int(events[i].value)
                    if events[i+4].value not in generate_voice:
                        events_without_voices.append(events[i])
                        events_without_voices.append(events[i+1])
                        events_without_voices.append(events[i+2])
                        events_without_voices.append(events[i+3])
                        events_without_voices.append(events[i+4])
            # print(events_without_voices, meter)
            # return
        if prompt and generate_voice is None:
            events = self.extract_events(prompt)
            words = [[]]
            for i in range(0, how_many_events):
                if i < len(events):
                    e = events[i]
                    words[0].append(self.event2word['{}_{}'.format(e.name, e.value)])
        else:
            words = []
            for _ in range(self.batch_size):
                ws = [self.event2word['Bar_None']]
                if 'chord' in self.checkpoint_path:
                    tempo_classes = [v for k, v in self.event2word.items() if 'Tempo Class' in k]
                    tempo_values = [v for k, v in self.event2word.items() if 'Tempo Value' in k]
                    chords = [v for k, v in self.event2word.items() if 'Chord' in k]
                    ws.append(self.event2word['Position_1/16'])
                    ws.append(np.random.choice(chords))
                    ws.append(self.event2word['Position_1/16'])
                    ws.append(np.random.choice(tempo_classes))
                    ws.append(np.random.choice(tempo_values))
                else:
                    # tempo_classes = [v for k, v in self.event2word.items() if 'Tempo Class' in k]
                    # tempo_values = [v for k, v in self.event2word.items() if 'Tempo Value' in k]
                    # ws.append(self.event2word['Position Bar_1'])
                    # ws.append(self.event2word['Position Quarter_1/16'])
                    # ws.append(np.random.choice(tempo_classes))
                    # ws.append(np.random.choice(tempo_values))
                    pass
                words.append(ws)
        # initialize mem
        batch_m = [np.zeros((self.mem_len, self.batch_size, self.d_model), dtype=np.float32) for _ in range(self.n_layer)]
        # generate
        original_length = len(words[0])
        initial_flag = 1
        current_generated_bar = 0
        while current_generated_bar < n_target_bar:
            # input
            if generate_voice is not None:
                # TODO resetowanie?
                # batch_m = [np.zeros((self.mem_len, self.batch_size, self.d_model), dtype=np.float32) for _ in range(self.n_layer)]
                original_length = len(words[0])
                temp_x = np.zeros((self.batch_size, original_length))
                for b in range(self.batch_size):
                    for z, t in enumerate(words[b]):
                        temp_x[b][z] = t
            elif initial_flag:
                temp_x = np.zeros((self.batch_size, original_length))
                for b in range(self.batch_size):
                    for z, t in enumerate(words[b]):
                        temp_x[b][z] = t
                initial_flag = 0
            else:
                temp_x = np.zeros((self.batch_size, 1))
                for b in range(self.batch_size):
                    temp_x[b][0] = words[b][-1]
            # prepare feed dict
            feed_dict = {self.x: temp_x}
            for m, m_np in zip(self.mems_i, batch_m):
                feed_dict[m] = m_np
            # model (prediction)
            _logits, _new_mem = self.sess.run([self.logits, self.new_mem], feed_dict=feed_dict)
            # sampling
            _logit = _logits[-1, 0]
            word = self.temperature_sampling(
                logits=_logit, 
                temperature=temperature,
                topk=topk)

            print(f'Gen: {utils_pk.word_to_event([int(word)], self.word2event)[0]},\t\tCurr {events_without_voices[current_position]},\t\t {events_without_voices[current_position+1]}')

            if generate_voice is not None:
                generated_event = utils_pk.word_to_event([int(word)], self.word2event)[0]
                add_from_original = False
                if generated_event.name == 'Position Bar':
                    last_generated_position[0] = int(generated_event.value)
                    words[0].append(word)
                elif generated_event.name == 'Position Quarter':
                    last_generated_position[1] = int(generated_event.value.split('/')[0])
                    words[0].append(word)
                elif generated_event.name == 'Note On' or generated_event.name == 'Note Duration':
                    words[0].append(word)
                elif generated_event.name == 'Bar':
                    add_from_original = True
                elif generated_event.name == 'Note Voice':
                    if int(generated_event.value) in generate_voice:
                        if events_without_voices[current_position].name == 'Position Bar':
                            # czy position wykracza
                            if (int(events_without_voices[current_position].value) < last_generated_position[0] or 
                                            (int(events_without_voices[current_position].value) == last_generated_position[0]
                                            and int(events_without_voices[current_position+1].value.split('/')[0]) < last_generated_position[1])):
                                words[0] = words[0][:-4]
                                add_from_original = True
                            else:
                                # re-new mem
                                batch_m = _new_mem
                                words[0].append(word)
                        elif events_without_voices[current_position].name == 'Bar':
                            if last_generated_position[0] <= meter:
                                # re-new mem
                                batch_m = _new_mem
                                words[0].append(word)
                            else:
                                words[0] = words[0][:-4]
                                add_from_original = True 
                    else:
                        words[0] = words[0][:-4]
                        add_from_original = True
                else:
                    print(f'FOKIN MISTAKE {generate_voice}')
                
                if add_from_original and current_position < len(events_without_voices):
                    if events_without_voices[current_position].name == 'Bar':
                        words[0].append(self.event2word['Bar_None'])
                        current_generated_bar += 1
                        current_position += 1
                    elif events_without_voices[current_position].name == 'Position Bar':
                        words[0].append(self.event2word['{}_{}'.format(events_without_voices[current_position].name, events_without_voices[current_position].value)])
                        words[0].append(self.event2word['{}_{}'.format(events_without_voices[current_position+1].name, events_without_voices[current_position+1].value)])
                        words[0].append(self.event2word['{}_{}'.format(events_without_voices[current_position+2].name, events_without_voices[current_position+2].value)])
                        words[0].append(self.event2word['{}_{}'.format(events_without_voices[current_position+3].name, events_without_voices[current_position+3].value)])
                        words[0].append(self.event2word['{}_{}'.format(events_without_voices[current_position+4].name, events_without_voices[current_position+4].value)])
                        current_position += 5
                # print(words[0])
            else:
                words[0].append(word)
                
                # if bar event (only work for batch_size=1)
                if word == self.event2word['Bar_None']:
                    current_generated_bar += 1
                # re-new mem
                batch_m = _new_mem
        # write
        utils_pk.write_midi(
            words=words[0],
            word2event=self.word2event,
            output_path=output_path,
            prompt_path=None)

    ########################################
    # prepare training data
    ########################################
    def prepare_data(self, midi_paths):
        # extract events
        all_events = []
        for path in midi_paths:
            try:
                events = self.extract_events(path)
                for trans in TRANSPONE:
                    transponed_events = []
                    for event in events:
                        if event.name == 'Note On':
                            new_event = utils_pk.Event(
                                name='Note On',
                                time=event.time, 
                                value=int(event.value + trans),
                                text='{}'.format(event.text))
                            transponed_events.append(new_event)
                        else:
                            transponed_events.append(event)
                    all_events.append(transponed_events)
            except Exception as e:
                print(f'ERROR: {path}, Exception: {e}')
        # event to word
        all_words = []
        for events in all_events:
            words = []
            try:
                for event in events:
                    e = '{}_{}'.format(event.name, event.value)
                    if e in self.event2word:
                        words.append(self.event2word[e])
                    else:
                        # OOV
                        if event.name == 'Note Velocity':
                            # replace with max velocity based on our training data
                            words.append(self.event2word['Note Velocity_128'])
                        elif event.name == "Position Bar" and int(event.value) > 8:
                            raise Exception(f"Too high meter.")
                        else:
                            raise Exception(f"Not known event.")
            except Exception as e:
                print(f'Exception: {e}, Event: {event}')
                continue
            all_words.append(words)
        # to training data
        self.group_size = 5
        segments = []
        for words in all_words:
            pairs = []
            for i in range(0, len(words)-self.x_len-1, self.x_len):
                x = words[i:i+self.x_len]
                y = words[i+1:i+self.x_len+1]
                pairs.append([x, y])
            pairs = np.array(pairs)
            # abandon the last
            for i in np.arange(0, len(pairs)-self.group_size, self.group_size*2):
                data = pairs[i:i+self.group_size]
                if len(data) == self.group_size:
                    segments.append(data)
        segments = np.array(segments)
        return segments

    ########################################
    # finetune
    ########################################
    def finetune(self, training_data, output_checkpoint_folder):
        # shuffle
        # tf.compat.v1.global_variables_initializer()
        # with tf.Session() as sess:
        #     sess.run(tf.compat.v1.global_variables_initializer())
        index = np.arange(len(training_data))
        np.random.shuffle(index)
        training_data = training_data[index]
        num_batches = len(training_data) // self.batch_size
        st = time.time()
        # bylo 200
        epoch_list = []
        gs_list = []
        loss_list = []
        time_list = []
        for e in range(50):
            total_loss = []
            for i in range(num_batches):
                segments = training_data[self.batch_size*i:self.batch_size*(i+1)]
                batch_m = [np.zeros((self.mem_len, self.batch_size, self.d_model), dtype=np.float32) for _ in range(self.n_layer)]
                for j in range(self.group_size):
                    batch_x = segments[:, j, 0, :]
                    batch_y = segments[:, j, 1, :]
                    # prepare feed dict
                    feed_dict = {self.x: batch_x, self.y: batch_y}
                    for m, m_np in zip(self.mems_i, batch_m):
                        feed_dict[m] = m_np
                    # run
                    _, gs_, loss_, new_mem_ = self.sess.run([self.train_op, self.global_step, self.avg_loss, self.new_mem], feed_dict=feed_dict)
                    if gs_ % 50 == 0:
                        epoch_list.append(e)
                        gs_list.append(gs_)
                        loss_list.append(loss_)
                        time_list.append(time.time()-st)

                    batch_m = new_mem_
                    total_loss.append(loss_)
                    print('>>> Epoch: {}, Step: {}, Loss: {:.5f}, Time: {:.2f}'.format(e, gs_, loss_, time.time()-st))
                    if gs_ % 1000 == 0:
                        self.saver.save(self.sess, '{}/model-{:03d}-{}-{:.3f}'.format(output_checkpoint_folder, e, gs_, np.mean(total_loss)))
                        with open('{}/learning_curve'.format(output_checkpoint_folder), 'wb+') as file:
                            pickle.dump((epoch_list, gs_list, loss_list, time_list), file)
            self.saver.save(self.sess, '{}/model-{:03d}-{:.3f}'.format(output_checkpoint_folder, e, np.mean(total_loss)))
            # stop
            if np.mean(total_loss) <= 0.1:
                break

    ########################################
    # close
    ########################################
    def close(self):
        self.sess.close()
