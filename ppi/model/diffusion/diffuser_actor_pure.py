import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from ppi.model.diffusion.diffuser_actor_utils.layers import (
    FFWRelativeSelfAttentionModule,
    FFWRelativeCrossAttentionModule,
    FFWRelativeSelfCrossAttentionModule
)
from ppi.model.diffusion.diffuser_actor_utils.layers import ParallelAttention
from ppi.model.diffusion.diffuser_actor_utils.position_encodings import (
    RotaryPositionEncoding3D,
    SinusoidalPosEmb
)
from pdb import set_trace

class DiffusionHeadPure(nn.Module):
# ablation for predict pure continuous actions or pure keyposes(keyframe actions)
    def __init__(self,
                 embedding_dim=120                                                                                                                               ,
                 num_attn_heads=8,
                 use_instruction=True,
                 rotation_parametrization='quat',
                 nhist=1,
                 lang_enhanced=False):
        super().__init__()
        self.use_instruction = use_instruction
        self.lang_enhanced = lang_enhanced
        if '6D' in rotation_parametrization:
            rotation_dim = 6  # continuous 6D
        else:
            rotation_dim = 4  # quaternion(we use this)

        # Encoders
        self.traj_encoder = nn.Linear(7, embedding_dim)
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(embedding_dim * nhist, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)

        # Attention from trajectory queries to language
        self.traj_lang_attention_left = nn.ModuleList([
            ParallelAttention(
                num_layers=1,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=False, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=False, apply_ffn=False
            )
        ])
        self.traj_lang_attention_right = nn.ModuleList([
            ParallelAttention(
                num_layers=1,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=False, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=False, apply_ffn=False
            )
        ])

        # Estimate attends to context (no subsampling)
        self.cross_attn = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=2, use_adaln=True
        )

        # Shared attention layers
        if not self.lang_enhanced:
            self.self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, num_layers=4, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads,
                num_self_attn_layers=4,
                num_cross_attn_layers=3,
                use_adaln=True
            )

        # Specific (non-shared) Output layers:
        # 1. Rotation
        self.rotation_proj_left = nn.Linear(embedding_dim, embedding_dim)
        if not self.lang_enhanced:
            self.rotation_self_attn_left = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.rotation_self_attn_left = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads, 2, 1, use_adaln=True
            )
        self.rotation_predictor_left = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, rotation_dim)
        )
        self.rotation_proj_right = nn.Linear(embedding_dim, embedding_dim)
        if not self.lang_enhanced:
            self.rotation_self_attn_right = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.rotation_self_attn_right = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads, 2, 1, use_adaln=True
            )
        self.rotation_predictor_right = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, rotation_dim)
        )

        # 2. Position
        self.position_proj_left = nn.Linear(embedding_dim, embedding_dim)
        if not self.lang_enhanced:
            self.position_self_attn_left = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.position_self_attn_left = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads, 2, 1, use_adaln=True
            )
        self.position_predictor_left = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3)
        )
        self.position_proj_right = nn.Linear(embedding_dim, embedding_dim)
        if not self.lang_enhanced:
            self.position_self_attn_right = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.position_self_attn_right = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads, 2, 1, use_adaln=True
            )
        self.position_predictor_right = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3)
        )

        # 3. Openess
        self.openess_predictor_left = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        self.openess_predictor_right = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, trajectory_left, trajectory_right, timestep,
            fixed_inputs):
        # set_trace()
        (pcd_coord, pcd_feat, lang_feat, state_feat, sampled_pcd_coord, sampled_pcd_feat) = fixed_inputs
        # Trajectory features(noisy actions)
        traj_feats_left = self.traj_encoder(trajectory_left)
        traj_feats_right = self.traj_encoder(trajectory_right)

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_feats_left.size(1), device=traj_feats_left.device)
        )[None].repeat(len(traj_feats_left), 1, 1)

        # set_trace()
        if self.use_instruction:
            traj_feats_left, _ = self.traj_lang_attention_left[0](
                seq1=traj_feats_left, seq1_key_padding_mask=None,
                seq2=lang_feat, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
            )

            traj_feats_right, _ = self.traj_lang_attention_right[0](
                seq1=traj_feats_right, seq1_key_padding_mask=None,
                seq2=lang_feat, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
            )
        traj_feats_left = traj_feats_left + traj_time_pos
        traj_feats_right = traj_feats_right + traj_time_pos

        traj_feats_left = einops.rearrange(traj_feats_left, 'b l c -> l b c')
        traj_feats_right = einops.rearrange(traj_feats_right, 'b l c -> l b c')
        pcd_feat = einops.rearrange(pcd_feat, 'b l c -> l b c')
        sampled_pcd_feat = einops.rearrange(sampled_pcd_feat, 'b l c -> l b c')
        state_feat = einops.rearrange(state_feat, 'b l c -> l b c')
        pos_pred_left, rot_pred_left, openess_pred_left, pos_pred_right, rot_pred_right, openess_pred_right = self.prediction_head(
            trajectory_left[..., :3], traj_feats_left,
            trajectory_right[..., :3], traj_feats_right,
            pcd_coord[..., :3], pcd_feat,
            timestep, state_feat,
            sampled_pcd_coord, sampled_pcd_feat,
            lang_feat
        )
        return ([torch.cat((pos_pred_left, rot_pred_left, openess_pred_left), -1)],[torch.cat((pos_pred_right, rot_pred_right, openess_pred_right), -1)])


    def prediction_head(self,
                trajectory_coord_left, traj_feats_left,
                trajectory_coord_right, traj_feats_right,
                pcd_coord, pcd_features,
                timesteps, state_feat,
                sampled_pcd_coord, sampled_pcd_features,
                lang_feat):
        # set_trace()
        # Diffusion timestep and state features
        time_embs = self.encode_denoising_timestep(
            timesteps, state_feat
        )

        # Positional embeddings
        rel_gripper_pos_left = self.relative_pe_layer(trajectory_coord_left)
        rel_gripper_pos_right = self.relative_pe_layer(trajectory_coord_right)
        rel_pcd_pos = self.relative_pe_layer(pcd_coord)
        sampled_rel_pcd_pos = self.relative_pe_layer(sampled_pcd_coord)
        
        # Cross attention from trajectory features to full pcd
        gripper_features_left = self.cross_attn(
            query=traj_feats_left,
            value=pcd_features,
            query_pos=rel_gripper_pos_left,
            value_pos=rel_pcd_pos,
            diff_ts=time_embs
        )[-1]

        gripper_features_right = self.cross_attn(
            query=traj_feats_right,
            value=pcd_features,
            query_pos=rel_gripper_pos_right,
            value_pos=rel_pcd_pos,
            diff_ts=time_embs
        )[-1]

        features = torch.cat([gripper_features_left, gripper_features_right, sampled_pcd_features], 0)
        rel_pos = torch.cat([rel_gripper_pos_left, rel_gripper_pos_right, sampled_rel_pcd_pos], 1)
        # self attention with sampled pcd features
        features = self.self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=lang_feat,
            context_pos=None
        )[-1]

        num_gripper_left = gripper_features_left.shape[0]
        num_gripper_right = gripper_features_right.shape[0]

        # Rotation head
        rotation_left = self.predict_rot_left(
            features, rel_pos, time_embs, 0, num_gripper_left, lang_feat
        )

        rotation_right = self.predict_rot_right(
            features, rel_pos, time_embs, num_gripper_left, num_gripper_left+num_gripper_right, lang_feat
        )

        # Position head
        position_left, position_features_left = self.predict_pos_left(
            features, rel_pos, time_embs, 0, num_gripper_left, lang_feat
        )
        position_right, position_features_right = self.predict_pos_right(
            features, rel_pos, time_embs, num_gripper_left, num_gripper_left+num_gripper_right, lang_feat
        )

        # Openess head from position head
        openess_left = self.openess_predictor_left(position_features_left)
        openess_right = self.openess_predictor_right(position_features_right)

        return position_left, rotation_left, openess_left, position_right, rotation_right, openess_right

    def encode_denoising_timestep(self, timestep, curr_gripper_features):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        # set_trace()
        time_feats = self.time_emb(timestep)

        curr_gripper_features = einops.rearrange(
            curr_gripper_features, "npts b c -> b npts c"
        )

        curr_gripper_features = curr_gripper_features.flatten(1)
        curr_gripper_feats = self.curr_gripper_emb(curr_gripper_features)
        return time_feats + curr_gripper_feats

    def predict_pos_left(self, features, rel_pos, time_embs, start_id, end_id,
                    instr_feats):
        # set_trace()
        position_features = self.position_self_attn_left(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]
        position_features = einops.rearrange(
            position_features[start_id:end_id], "npts b c -> b npts c"
        )
        position_features = self.position_proj_left(position_features)  # (B, N, C)
        position = self.position_predictor_left(position_features)
        return position, position_features
    
    def predict_pos_right(self, features, rel_pos, time_embs, start_id, end_id,
                    instr_feats):
        # set_trace()
        position_features = self.position_self_attn_right(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]
        position_features = einops.rearrange(
            position_features[start_id:end_id], "npts b c -> b npts c"
        )
        position_features = self.position_proj_right(position_features)  # (B, N, C)
        position = self.position_predictor_right(position_features)
        return position, position_features

    def predict_rot_left(self, features, rel_pos, time_embs, start_id, end_id,
                    instr_feats):
        # set_trace()
        rotation_features = self.rotation_self_attn_left(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]
        rotation_features = einops.rearrange(
            rotation_features[start_id:end_id], "npts b c -> b npts c"
        )
        rotation_features = self.rotation_proj_left(rotation_features)  # (B, N, C)
        rotation = self.rotation_predictor_left(rotation_features)
        return rotation
    
    def predict_rot_right(self, features, rel_pos, time_embs, start_id, end_id,
                    instr_feats):
        rotation_features = self.rotation_self_attn_right(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]
        rotation_features = einops.rearrange(
            rotation_features[start_id:end_id], "npts b c -> b npts c"
        )
        rotation_features = self.rotation_proj_right(rotation_features)  # (B, N, C)
        rotation = self.rotation_predictor_right(rotation_features)
        return rotation