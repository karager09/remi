import chord_recognition
import numpy as np
import miditoolkit
import copy

# parameters for input
MIN_DURATION = 1/8

DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32+1, dtype=np.int)
DEFAULT_DURATION_BINS = np.arange(0, 4.01, MIN_DURATION, dtype=float)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# parameters for output
DEFAULT_RESOLUTION = 480

# define "Item" for general storage
class Item(object):
    def __init__(self, name, start, end, velocity, pitch):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch

    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, pitch={})'.format(
            self.name, self.start, self.end, self.velocity, self.pitch)

# read notes and tempo changes from midi (assume there is only one track)
def read_items(midi_obj):
    # midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    ticks_per_beat = midi_obj.ticks_per_beat
    # note
    note_items = []
    notes = midi_obj.instruments[0].notes
    if 1 < len(midi_obj.instruments):
        notes += midi_obj.instruments[1].notes
    notes.sort(key=lambda x: (x.start, x.pitch))
    for note in notes:
        note_items.append(Item(
            name='Note', 
            start=note.start/ticks_per_beat, 
            end=note.end/ticks_per_beat, 
            velocity=note.velocity, 
            pitch=note.pitch))
    note_items.sort(key=lambda x: x.start)
    # tempo
    tempo_items = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append(Item(
            name='Tempo',
            start=tempo.time/ticks_per_beat,
            end=None,
            velocity=None,
            pitch=int(tempo.tempo)))
    tempo_items.sort(key=lambda x: x.start)
    # expand to all beat
    max_tick = tempo_items[-1].start
    existing_ticks = {item.start: item.pitch for item in tempo_items}
    wanted_ticks = np.arange(0, max_tick+1, 1)
    output = []
    for tick in wanted_ticks:
        if tick in existing_ticks:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=existing_ticks[tick]))
        else:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=output[-1].pitch))
    tempo_items = output
    return note_items, tempo_items

# quantize items
def quantize_items(items):
    for item in items:
        if item.start % MIN_DURATION != 0:
            diff = item.start % MIN_DURATION
            if diff <= MIN_DURATION/2:
                item.start -= diff
                item.end -= diff
            else:
                item.start = item.start - diff + MIN_DURATION
                item.end = item.end - diff + MIN_DURATION
    return items      

# extract chord
def extract_chords(items):
    method = chord_recognition.MIDIChord()
    chords = method.extract(notes=items)
    output = []
    for chord in chords:
        output.append(Item(
            name='Chord',
            start=chord[0],
            end=chord[1],
            velocity=None,
            pitch=chord[2].split('/')[0]))
    return output

# group items
def group_items(items, max_time, ticks_per_beat, time_signature_changes=None):
    items.sort(key=lambda x: x.start)
    downbeats = []
    for time_signature, next_time_signature in zip(time_signature_changes, (time_signature_changes + [None])[1:]):
        quarter_per_bar =  int(4 / time_signature.denominator * time_signature.numerator)
        start_time = int(time_signature.time / ticks_per_beat)
        if next_time_signature != None:
            end_time = int(next_time_signature.time / ticks_per_beat)
        else:
            end_time = int(max_time + quarter_per_bar)

        # print(start_time, end_time, quarter_per_bar, time_signature, next_time_signature, sep='\n')
        downbeats.extend(range(start_time, end_time, quarter_per_bar))
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = []
        for item in items:
            if (item.start >= db1) and (item.start < db2):
                # if item.start != db1 and item.name == 'Tempo':
                    # leave tempo items only on the beginning of bars
                    # continue
                insiders.append(item)
        overall = [db1] + insiders + [db2]
        groups.append(overall)
    return groups

# define "Event" for event storage
class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return 'Event(name={}, time={}, value={}, text={})'.format(
            self.name, self.time, self.value, self.text)

# item to event
def item2event(groups):
    events = []
    n_downbeat = 0
    ticks_per_quarter = int(1/MIN_DURATION)
    for i in range(len(groups)):
        if 'Note' not in [item.name for item in groups[i][1:-1]]:
            continue
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        events.append(Event(
            name='Bar',
            time=None, 
            value=None,
            text='{}'.format(n_downbeat)))
        for item in groups[i][1:-1]:
            # position
            position_in_bar = int(item.start - bar_st + 1)
            position_in_quarter = int((item.start % 1) * ticks_per_quarter + 1)
            events.append(Event(
                name='Position Bar', 
                time=item.start,
                value='{}'.format(position_in_bar),
                text='{}'.format(item.start)))
            events.append(Event(
                name='Position Quarter', 
                time=item.start,
                value='{}/{}'.format(position_in_quarter, ticks_per_quarter),
                text='{}'.format(item.start)))
            if item.name == 'Note':
                # velocity
                velocity_index = np.searchsorted(
                    DEFAULT_VELOCITY_BINS, 
                    item.velocity, 
                    side='right') - 1
                events.append(Event(
                    name='Note Velocity',
                    time=item.start, 
                    value=DEFAULT_VELOCITY_BINS[velocity_index],
                    text='{}/{}'.format(item.velocity, DEFAULT_VELOCITY_BINS[velocity_index])))
                # pitch
                events.append(Event(
                    name='Note On',
                    time=item.start, 
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
                # duration
                duration = item.end - item.start
                if duration > 4:
                    duration = int(4 * (1/MIN_DURATION)) # max duration is 4 quarters
                else:
                    duration = int(duration * ticks_per_quarter)
                events.append(Event(
                    name='Note Duration',
                    time=item.start,
                    value=duration,
                    text='{}/{}'.format(duration, ticks_per_quarter)))
            elif item.name == 'Chord':
                events.append(Event(
                    name='Chord', 
                    time=item.start,
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
            elif item.name == 'Tempo':
                tempo = item.pitch
                if tempo in DEFAULT_TEMPO_INTERVALS[0]:
                    tempo_style = Event('Tempo Class', item.start, 'slow', None)
                    tempo_value = Event('Tempo Value', item.start, 
                        tempo-DEFAULT_TEMPO_INTERVALS[0].start, None)
                elif tempo in DEFAULT_TEMPO_INTERVALS[1]:
                    tempo_style = Event('Tempo Class', item.start, 'mid', None)
                    tempo_value = Event('Tempo Value', item.start, 
                        tempo-DEFAULT_TEMPO_INTERVALS[1].start, None)
                elif tempo in DEFAULT_TEMPO_INTERVALS[2]:
                    tempo_style = Event('Tempo Class', item.start, 'fast', None)
                    tempo_value = Event('Tempo Value', item.start, 
                        tempo-DEFAULT_TEMPO_INTERVALS[2].start, None)
                elif tempo < DEFAULT_TEMPO_INTERVALS[0].start:
                    tempo_style = Event('Tempo Class', item.start, 'slow', None)
                    tempo_value = Event('Tempo Value', item.start, 0, None)
                elif tempo > DEFAULT_TEMPO_INTERVALS[2].stop:
                    tempo_style = Event('Tempo Class', item.start, 'fast', None)
                    tempo_value = Event('Tempo Value', item.start, 59, None)
                events.append(tempo_style)
                events.append(tempo_value)     
    return events

#############################################################################################
# WRITE MIDI
#############################################################################################
def make_map():
    list_of_events = []
    list_of_events.append('Bar_None')
    for i in range(1, 7):
        list_of_events.append(f'Position Bar_{i}')
    for i in range(1, 9):
        list_of_events.append(f'Position Quarter_{i}/8')
    for i in DEFAULT_VELOCITY_BINS:
        list_of_events.append(f'Note Velocity_{i}')
    for i in DEFAULT_DURATION_BINS:
        list_of_events.append(f'Note Duration_{int(i/MIN_DURATION)}')
    for i in range(0, 128):
        list_of_events.append(f'Note On_{i}')
    for i in ['slow', 'mid', 'fast']:
        list_of_events.append(f'Tempo Class_{i}')
    for i in range(0, 60):
        list_of_events.append(f'Tempo Value_{i}')
    # list_of_events
    iterator = 1
    event2word = {}
    word2event = {}
    for name in list_of_events:
        event2word[name] = iterator
        word2event[iterator] = name
        iterator += 1
    return event2word, word2event

def word_to_event(words, word2event):
    events = []
    for word in words:
        event_name, event_value = word2event.get(word).split('_')
        events.append(Event(event_name, None, event_value, None))
    return events

def event_to_word(events, event2word):
    words = []
    for event in events:
        event_string = f'{event.name}_{event.value}'
        word = event2word.get(event_string)
        if word is None:
            print(f'Dont exists: {event_string}')
            pass
        else:
            words.append(word)
    return words

def write_midi(words, word2event, output_path, prompt_path=None):
    events = word_to_event(words, word2event)
    # get downbeat and note (no time)
    temp_notes = []
    temp_chords = []
    temp_tempos = []
    last_meter = 0
    for i in range(len(events)-3):
        if events[i].name == 'Bar':
            if 'Bar' in temp_notes:
                last_bar_index = max(loc for loc, val in enumerate(temp_notes) if val == 'Bar')
                temp_notes[last_bar_index+1] = last_meter
            temp_notes.append('Bar')
            temp_notes.append(last_meter)
            # temp_chords.append('Bar')
            if 'Bar' in temp_tempos:
                last_bar_index = max(loc for loc, val in enumerate(temp_tempos) if val == 'Bar')
                temp_tempos[last_bar_index+1] = last_meter
            temp_tempos.append('Bar')
            temp_tempos.append(last_meter)
        elif events[i].name == 'Position Bar' and \
            events[i+1].name == 'Position Quarter' and \
            events[i+2].name == 'Note Velocity' and \
            events[i+3].name == 'Note On' and \
            events[i+4].name == 'Note Duration':
            # start time and end time from position
            last_meter = int(events[i].value)
            position_in_bar = int(events[i].value) - 1
            position_in_quarter = int(events[i+1].value.split('/')[0]) - 1
            position = position_in_bar + position_in_quarter * MIN_DURATION
            # velocity
            velocity = int(events[i+2].value)
            # pitch
            pitch = int(events[i+3].value)
            # duration
            duration = int(events[i+4].value)
            duration = duration * MIN_DURATION
            # adding
            temp_notes.append([position, velocity, pitch, duration])
        # elif events[i].name == 'Position' and events[i+1].name == 'Chord':
        #     position = int(events[i].value.split('/')[0]) - 1
        #     temp_chords.append([position, events[i+1].value])
        elif events[i].name == 'Position Bar' and \
            events[i+1].name == 'Position Quarter' and \
            events[i+2].name == 'Tempo Class' and \
            events[i+3].name == 'Tempo Value':
            last_meter = int(events[i].value)
            position_in_bar = int(events[i].value) - 1
            position_in_quarter = int(events[i+1].value.split('/')[0]) - 1
            position = position_in_bar + position_in_quarter * MIN_DURATION
            if events[i+2].value == 'slow':
                tempo = DEFAULT_TEMPO_INTERVALS[0].start + int(events[i+3].value)
            elif events[i+2].value == 'mid':
                tempo = DEFAULT_TEMPO_INTERVALS[1].start + int(events[i+3].value)
            elif events[i+2].value == 'fast':
                tempo = DEFAULT_TEMPO_INTERVALS[2].start + int(events[i+3].value)
            temp_tempos.append([position, tempo])
    # get specific time for notes
    notes = []
    current_bar = -1
    time_to_current_bar = 0
    current_meter = 0
    for note in temp_notes:
        if note == 'Bar':
            current_bar += 1
            time_to_current_bar += current_meter
        elif type(note) is not list:
            current_meter = int(note)
        else:
            position, velocity, pitch, duration = note
            absolute_position_st = (time_to_current_bar + position) * DEFAULT_RESOLUTION
            absolute_position_et = (time_to_current_bar + position + duration) * DEFAULT_RESOLUTION
            notes.append(miditoolkit.Note(velocity, pitch, int(absolute_position_st), int(absolute_position_et)))
    # get specific time for chords
    # if len(temp_chords) > 0:
    #     chords = []
    #     current_bar = 0
    #     for chord in temp_chords:
    #         if chord == 'Bar':
    #             current_bar += 1
    #         else:
    #             position, value = chord
    #             # position (start time)
    #             current_bar_st = current_bar * ticks_per_bar
    #             current_bar_et = (current_bar + 1) * ticks_per_bar
    #             flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
    #             st = flags[position]
    #             chords.append([st, value])
    # get specific time for tempos
    tempos = []
    meter = []
    # teraz zaczyna sie od 'Bar_None'
    current_bar = -1
    time_to_current_bar = 0
    current_meter = 0
    for tempo in temp_tempos:
        if tempo == 'Bar':
            current_bar += 1
            time_to_current_bar += current_meter
        elif type(tempo) is not list:
            current_meter = int(tempo)
            if len(meter) == 0 or meter[-1][1] != current_meter:
                meter.append([int(time_to_current_bar * DEFAULT_RESOLUTION), current_meter])
        else:
            position, value = tempo
            absolute_position_st = (time_to_current_bar + position) * DEFAULT_RESOLUTION
            tempos.append([int(absolute_position_st), int(value)])
    # write
    if prompt_path:
        # TODO
        midi = miditoolkit.midi.parser.MidiFile(prompt_path)
        #
        last_time = DEFAULT_RESOLUTION * 4 * 4
        # note shift
        for note in notes:
            note.start += last_time
            note.end += last_time
        midi.instruments[0].notes.extend(notes)
        # tempo changes
        temp_tempos = []
        for tempo in midi.tempo_changes:
            if tempo.time < DEFAULT_RESOLUTION*4*4:
                temp_tempos.append(tempo)
            else:
                break
        for st, bpm in tempos:
            st += last_time
            temp_tempos.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = temp_tempos
        # write chord into marker
        if len(temp_chords) > 0:
            for c in chords:
                midi.markers.append(
                    miditoolkit.midi.containers.Marker(text=c[1], time=c[0]+last_time))
    else:
        midi = miditoolkit.midi.parser.MidiFile()
        midi.ticks_per_beat = DEFAULT_RESOLUTION
        # write instrument
        inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
        inst.notes = notes
        midi.instruments.append(inst)
        # write tempo
        tempo_changes = []
        for st, bpm in tempos:
            tempo_changes.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = tempo_changes
        # write meter
        meter_changes = []
        for st, new_meter in meter:
            meter_changes.append(miditoolkit.midi.containers.TimeSignature(time=st, numerator=new_meter, denominator=4))
        midi.time_signature_changes = meter_changes
        # write chord into marker
        # if len(temp_chords) > 0:
        #     for c in chords:
        #         midi.markers.append(
        #             miditoolkit.midi.containers.Marker(text=c[1], time=c[0]))
    # write
    midi.dump(output_path)
