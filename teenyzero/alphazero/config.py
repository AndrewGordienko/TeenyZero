from teenyzero.alphazero.runtime import get_runtime_profile


PROFILE = get_runtime_profile()

INPUT_HISTORY_LENGTH = PROFILE.input_history_length
PIECE_PLANES_PER_POSITION = PROFILE.piece_planes_per_position
AUX_PLANES = PROFILE.aux_planes
INPUT_PLANES = PROFILE.input_planes
INPUT_SHAPE = PROFILE.input_shape

MODEL_VERSION = PROFILE.model_version
MODEL_RES_BLOCKS = PROFILE.model_res_blocks
MODEL_CHANNELS = PROFILE.model_channels
POLICY_HEAD_CHANNELS = PROFILE.policy_head_channels
VALUE_HEAD_HIDDEN = PROFILE.value_head_hidden

REPLAY_ENCODER_VERSION = PROFILE.replay_encoder_version
