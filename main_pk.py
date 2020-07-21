from model_pk import PopMusicTransformer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    print("Starting..")
    # declare model
    model = PopMusicTransformer(
        checkpoint='REMI-finetune-pk-len256-2from3ins_0,1err_anymeter_v2',
        model_name="model-005-64000-0.696",
        is_training=False)
    # model = PopMusicTransformer(
    #     checkpoint='REMI-finetune-pk-chorales-transposed',
    #     model_name="model-004-16000-0.349",
    #     is_training=False)
    print("Loading complete")

    # generate from scratch
    # model.generate(
    #     n_target_bar=16,
    #     temperature=1.1,
    #     topk=5,
    #     output_path='./result/chorales/transposed/from_scratch_pk.mid',
    #     prompt=None)
    # print("From scratch complete")

    # generate continuation
    # output = './result/chorales/transposed/continuation_pk.mid'
    output = './result/mine/continuation_pk.mid'
    model.generate(
        n_target_bar=16,
        temperature=1.1,
        topk=5,
        output_path=output,
        prompt=r"D:\Studia\magisterka\Pliki midi\Classical Archives - The Greats (MIDI Library)\Classical Archives - The Greats (MIDI)\Bach\Bwv001- 400 Chorales\003706b_.mid")
    # model.generate(
    #     n_target_bar=16,
    #     temperature=1.2,
    #     topk=5,
    #     output_path=output,
    #     prompt=r"D:\Studia\magisterka\Pliki midi\Classical Archives - The Greats (MIDI Library)\Classical Archives - The Greats (MIDI)\Beethoven\Piano Sonatas\Piano Sonata n14 op27  ''Moonlight''.mid")
    print("Continuation complete")

    # close model
    model.close()

if __name__ == '__main__':
    main()
