from model_pk import PopMusicTransformer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    print("Starting..")
    # declare model
    model = PopMusicTransformer(
        checkpoint='REMI-finetune-pk-len256-2from3ins_0,1err',
        model_name="model-022-0.570",
        is_training=False)
    print("Loading complete")

    # generate from scratch
    model.generate(
        n_target_bar=16,
        temperature=1.1,
        topk=5,
        output_path='./result/from_scratch_pk.midi',
        prompt=None)
    print("From scratch complete")

    # generate continuation
    model.generate(
        n_target_bar=16,
        temperature=1.1,
        topk=5,
        output_path='./result/continuation_pk.midi',
        prompt='./data/evaluation/000.midi')
    print("Continuation complete")

    # close model
    model.close()

if __name__ == '__main__':
    main()
