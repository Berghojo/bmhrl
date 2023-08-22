from scripts.train_rl_captioning_module import train_rl_cap
import argparse
from pprint import pprint
from utilities.config_constructor import Config
def create_config():
    '''
            Note, that the arguments are shared for both train_cap and train_prop that leads to the
            situation in which an argument is defined but unused (--word_emb_caps for train_prop case).
        '''
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument('--rl_high_level_enc_d', type=int, default=256, help='rl enc dimensions')
    parser.add_argument('--rl_low_level_enc_d', type=int, default=512, help='rl enc dimensions')
    parser.add_argument('--rl_worker_lstm', type=int, default=1024, help='rl worker dimensions')
    # TODO paper had 256 for manager lstm - doesnt work with bidirectional encoder  goin up to 512?
    parser.add_argument('--rl_manager_lstm', type=int, default=512, help='rl manager dimensions')
    parser.add_argument('--rl_goal_d', type=int, default=64, help='rl goal dimensions')
    parser.add_argument('--rl_attn_d', type=int, default=512, help='rl attention dimensions')
    parser.add_argument('--rl_critic_path', type=str, default='./data/models/critic.cp',
                        help='ddpg critic checkpoint path')
    parser.add_argument('--rl_critic_score_threshhold', type=float, default=0.25,
                        help='critic threshhold after softmax for labelling segments')
    parser.add_argument('--rl_dropout', type=float, default=0.1, help='rl dropout')

    parser.add_argument('--rl_gamma_worker', type=float, default=0.8, help='reward diminishing constant')
    parser.add_argument('--rl_gamma_manager', type=float, default=0.8, help='reward diminishing constant')

    parser.add_argument('--rl_pretrained_model_dir', type=str, help="pretrained rl model to use")
    parser.add_argument('--rl_train_worker', type=bool, default=True, help="train worker or manager")
    parser.add_argument('--rl_warmstart_epochs', type=int, default=0,
                        help="Epochs trained via wamrstart before starting the agent")
    parser.add_argument('--rl_projection_d', type=int, default=512, help='dimension for common projection space')
    parser.add_argument('--rl_att_heads', type=int, default=4, help='#attention heads')
    parser.add_argument('--rl_att_layers', type=int, default=2, help='#attention layers')

    parser.add_argument('--rl_reward_weight_worker', type=int, default=1, help='weighting rewards additionally')
    parser.add_argument('--rl_reward_weight_manager', type=int, default=2, help='weighting rewards additionally')

    # Feed Forward intermediate dims
    parser.add_argument('--rl_ff_c', type=int, default=2048, help='caption FF Layer dim')
    parser.add_argument('--rl_ff_v', type=int, default=1024, help='video FF Layer dim')
    parser.add_argument('--rl_ff_a', type=int, default=512, help='audio FF Layer dim')

    # Use baseline to stabilize training
    parser.add_argument('--rl_stabilize', type=bool, default=False, help='stabilize rl training')

    parser.add_argument('--rl_value_function_lr', type=float, default=1e-4, help='value function lr')
    parser.add_argument('--rl_cap_warmstart_lr', type=float, default=1e-4, help='warmstart captioning lr')
    parser.add_argument('--rl_cap_lr', type=float, default=0.0005, help='warmstart captioning lr')
    parser.add_argument('--mode', type=str, default='DETR', choices=['DETR', 'BMHRL', 'BM', 'AHRL', 'VHRL', 'verbose', 'eval'],
                        help="Ablation study modes")
    parser.add_argument('--scorer', type=str, default='BLEU', choices=['CIDER', 'METEOR', 'BLEU'])

    ## Critic

    parser.add_argument('--train_csv_path', type=str, default='./data/Critic/critic_training.csv')

    ## DATA
    # paths to the precalculated train meta files
    parser.add_argument('--vatex_meta_path', type=str, default='./data/vatex_val.csv')
    parser.add_argument('--train_meta_path', type=str, default='./data/train.csv')
    parser.add_argument('--val_1_meta_path', type=str, default='./data/val_1.csv')
    parser.add_argument('--val_2_meta_path', type=str, default='./data/val_2.csv')

    parser.add_argument('--msrvtt_meta_path', type=str, default='./data/msrvtt_val.csv')
    parser.add_argument('--segmentation_vocab_path', type=str, default='./data/combined_captions.csv')

    parser.add_argument('--modality', type=str, default='audio_video',
                        choices=['audio', 'video', 'audio_video'],
                        help='modality to use. if audio_video both audio and video are used')
    parser.add_argument('--video_feature_name', type=str, default='i3d')
    parser.add_argument('--audio_feature_name', type=str, default='vggish')
    parser.add_argument('--video_features_path', type=str,
                        default='./data/i3d_25fps_stack64step64_2stream_npy/')
    parser.add_argument('--audio_features_path', type=str,
                        default='./data/vggish_npy/')
    parser.add_argument('--d_vid', type=int, default=1024, help='raw feature dimension')
    parser.add_argument('--d_aud', type=int, default=128, help='raw feature dimension')
    parser.add_argument('--word_emb_caps', default='glove.840B.300d', type=str,
                        help='Embedding code name from torchtext.vocab.Vocab')
    parser.add_argument('--unfreeze_word_emb', dest='unfreeze_word_emb', action='store_true',
                        default=False, help='Whether to finetune the pre-trained text embeddings')
    parser.add_argument('--feature_timespan_in_fps', type=int, default=64,
                        help='how many fps the input features will temporally cover')
    parser.add_argument('--fps_at_extraction', type=int, default=25,
                        help='how many fps were used at feature extraction')
    parser.add_argument('--audio_feature_timespan', type=float,
                        default=0.96, help='audio feature timespan')
    parser.add_argument('--train_json_path', type=str, default='./data/train.json')
    parser.add_argument('--train_segment_json_path', type=str, default='./data/CharadeCaptions/charades_captions.json')

    parser.add_argument('--device_ids', type=int, nargs='+', default=[0], help='separated by a whitespace')
    parser.add_argument('--start_token', type=str, default='<s>', help='starting token')
    parser.add_argument('--end_token', type=str, default='</s>', help='ending token')
    parser.add_argument('--pad_token', type=str, default='<blank>', help='padding token')
    parser.add_argument('--max_len', type=int, default=30, help='maximum size of 1by1 prediction')
    parser.add_argument('--min_freq_caps', type=int, default=1,
                        help='a word should appear min_freq times in train dataset to be in the vocab')

    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.999], help='betas in adam')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps in adam')

    parser.add_argument('--lr', type=float, default=1e-4, help='lr (if scheduler is constant)')
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--B', type=int, default=32, help='batch size per device')
    parser.add_argument('--inf_B_coeff', type=int, default=2,
                        help='The batch size on inference will be inf_B_coeff times B arg')
    parser.add_argument('--epoch_num', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--one_by_one_starts_at', type=int, default=10,
                        help='# of epochs to skip before starting 1-by-1 validation (saves time)')
    parser.add_argument('--early_stop_after', type=int, default=30,
                        help='number of epochs to wait for best metric to change before stopping')
    parser.add_argument(
        '--smoothing', type=float, default=0.7,
        help='smoothing coeff (= 0 cross ent loss, more -- stronger smoothing) must be in [0, 1]'
    )
    parser.add_argument('--grad_clip', type=float, help='max grad norm for gradients')

    parser.add_argument('--pad_audio_feats_up_to', type=int, default=800,
                        help='max feature length to pad other features to')
    parser.add_argument('--pad_video_feats_up_to', type=int, default=300,
                        help='max feature length to pad other features to')
    parser.add_argument('--nms_tiou_thresh', type=float, help='non-max suppression objectness thr')
    parser.add_argument('--log_dir', type=str, default='./log/')

    ## EVALUATION
    parser.add_argument('--prop_pred_path', type=str, help='path to a .json file with prop preds')
    parser.add_argument('--avail_mp4_path', type=str, default='./data/available_mp4.txt',
                        help='list of available videos')
    parser.add_argument('--reference_paths', type=str, nargs='+',
                        default=['./data/val_1_no_missings.json', './data/val_2_no_missings.json', './data/vatex_no_missings.json', './data/msrvtt_no_missings.json'],
                        help='reference paths for 1-by-1 validation')
    parser.add_argument('--tIoUs', type=float, default=[0.3, 0.5, 0.7, 0.9], nargs='+',
                        help='thresholds for tIoU to be used for 1-by-1 validation')
    parser.add_argument(
        '--max_prop_per_vid', type=int, default=100,
        help='max number of proposals to take into considetation in 1-by-1 validation'
    )
    parser.add_argument('--val_prop_meta_path', type=str, help='Only used in eval_on_learnd_props')

    ## MODEL
    parser.add_argument(
        '--d_model', type=int, default=1024,
        help='the internal space in the mullti-headed attention (when input dims of Q, K, V differ)')

    parser.add_argument(
        '--d_model_video', type=int,
        help='If use_linear_embedder is true, this is going to be the d_model size for video model'
    )
    parser.add_argument(
        '--d_model_audio', type=int,
        help='If use_linear_embedder is true, this is going to be the d_model size for audio model'
    )
    parser.add_argument(
        '--d_model_caps', type=int, default=300,
        help='hidden size of the crossmodal decoder (caption tokens are mapped into this dim)'
    )
    parser.add_argument(
        '--use_linear_embedder', dest='use_linear_embedder', action='store_true', default=False,
        help='Whether to include a dense layer between the raw features and input to the model'
    )

    parser.add_argument('--dout_p', type=float, default=0.1, help='dropout probability: in [0, 1]')

    parser.add_argument('--scheduler', type=str, default='constant',
                        choices=['constant', 'reduce_on_plateau'], help='lr scheduler')

    parser.add_argument('--procedure', type=str, default='train_rl_cap',
                        choices=['train_rl_cap'])

    ## DEBUGGING
    parser.add_argument('--debug', dest='debug', action='store_true', default=False,
                        help='runs test() instead of main()')
    parser.add_argument('--dont_log', dest='to_log', action='store_false',
                        help='Prevent logging in the experiment.')

    parser.set_defaults(to_log=True)

    args = parser.parse_args()
    pprint(vars(args))
    cfg = Config(args)
    #print(cfg.vat_meta_path)
    return cfg

if __name__ == "__main__":
    print('creating cfg')
    cfg = create_config()
    print('start')
    train_rl_cap(cfg)
