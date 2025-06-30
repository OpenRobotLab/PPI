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


class DiffusionHeadPPIReal(nn.Module):
    #给了监督但是condition on pointflow

    def __init__(self,
                 embedding_dim=120                                                                                                                               ,
                 num_attn_heads=8,
                 use_instruction=True,
                 rotation_parametrization='quat',
                 nhist=1,
                 lang_enhanced=False,
                 horizon_keyframe=2,
                 horizon_continuous=3):
        super().__init__()
        self.use_instruction = use_instruction
        self.lang_enhanced = lang_enhanced
        self.horizon_keyframe = horizon_keyframe
        self.horizon_continuous = horizon_continuous
        if '6D' in rotation_parametrization:
            rotation_dim = 6  # continuous 6D
        else:
            rotation_dim = 4  # quaternion(we use this)

        # Encoders
        self.traj_encoder = nn.Linear(7, embedding_dim)
        self.point_flow_encoder = nn.Linear(3, embedding_dim)
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
        self.point_flow_time_emb = SinusoidalPosEmb(embedding_dim)
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
        self.point_flow_lang_attention = nn.ModuleList([
            ParallelAttention(
                num_layers=1,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=False, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=False, apply_ffn=False
            )
        ])
        self.point_flow_lang_attention = nn.ModuleList([
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
        
        self.cross_attn_pointflow = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=2, use_adaln=True
        )

        # Shared attention layers
        if not self.lang_enhanced:
            self.self_attn_keyframe = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, num_layers=4, use_adaln=True
            )
            self.self_attn_continuous = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, num_layers=4, use_adaln=True
            )
            self.self_attn_point_flow = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, num_layers=4, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.self_attn_keyframe = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads,
                num_self_attn_layers=4,
                num_cross_attn_layers=3,
                use_adaln=True
            )
            self.self_attn_continuous = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads,
                num_self_attn_layers=4,
                num_cross_attn_layers=3,
                use_adaln=True
            )
            self.self_attn_point_flow = FFWRelativeSelfCrossAttentionModule(
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
        
        self.position_proj_point_flow = nn.Linear(embedding_dim, embedding_dim)
        if not self.lang_enhanced:
            self.position_self_attn_point_flow = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.position_self_attn_point_flow = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads, 2, 1, use_adaln=True
            )
        self.position_predictor_point_flow = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 12)
        )

        # 3. Openess
        self.openess_predictor_left_kf = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        self.openess_predictor_left_cn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        self.openess_predictor_right_kf = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        self.openess_predictor_right_cn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, trajectory_left, trajectory_right, timestep,
            fixed_inputs):
        # set_trace()
        (context_coord, context_feat, lang_feat, state_feat, pn_coord, pn_feat, pointflow_feat, pointflow_coords) = fixed_inputs
        # Trajectory features
        traj_feats_left = self.traj_encoder(trajectory_left) 
        traj_feats_right = self.traj_encoder(trajectory_right)

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_feats_left.size(1), device=traj_feats_left.device)
        )[None].repeat(len(traj_feats_left), 1, 1)
        
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

        context_feat = einops.rearrange(context_feat, 'b l c -> l b c') 
        pn_feat = einops.rearrange(pn_feat, 'b l c -> l b c')
        state_feat = einops.rearrange(state_feat, 'b l c -> l b c')
        pointflow_feat = einops.rearrange(pointflow_feat, 'b l c -> l b c')
        
        pos_pred_left, rot_pred_left, openess_pred_left, pos_pred_right, rot_pred_right, openess_pred_right, position_point_flow = self.prediction_head(
            trajectory_left[..., :3], traj_feats_left,
            trajectory_right[..., :3], traj_feats_right,
            context_coord[..., :3], context_feat,
            timestep, state_feat,
            pn_coord, pn_feat,
            lang_feat, pointflow_feat, pointflow_coords
        )
        return ([torch.cat((pos_pred_left, rot_pred_left, openess_pred_left), -1)],[torch.cat((pos_pred_right, rot_pred_right, openess_pred_right), -1)],[position_point_flow])

    def prediction_head(self,
                trajectory_coord_left, traj_feats_left,
                trajectory_coord_right, traj_feats_right,
                context_pcd, context_features,
                timesteps, state_feat,
                sampled_context_pcd, sampled_context_features,
                lang_feat,
                pointflow_feat, pointflow_coords):

        # Diffusion timestep
        time_embs = self.encode_denoising_timestep(
            timesteps, state_feat
        )
        

        # Positional embeddings
        rel_gripper_pos_left = self.relative_pe_layer(trajectory_coord_left)
        rel_gripper_pos_right = self.relative_pe_layer(trajectory_coord_right)
        
        rel_context_pos = self.relative_pe_layer(context_pcd)
        rel_initial_point_flow = self.relative_pe_layer(pointflow_coords)
        sampled_rel_context_pos = self.relative_pe_layer(sampled_context_pcd)
        # Cross attention from gripper to full context
        gripper_features_left = self.cross_attn(
            query=traj_feats_left,
            value=context_features,
            query_pos=rel_gripper_pos_left,
            value_pos=rel_context_pos,
            diff_ts=time_embs
        )[-1]
        gripper_features_right = self.cross_attn(
            query=traj_feats_right,
            value=context_features,
            query_pos=rel_gripper_pos_right,
            value_pos=rel_context_pos,
            diff_ts=time_embs
        )[-1]
        
        point_flow_features = self.cross_attn_pointflow(
            query=pointflow_feat,
            value=context_features,
            query_pos=rel_initial_point_flow,
            value_pos=rel_context_pos,
            diff_ts=time_embs
        )[-1]
        
        features_point_flow = torch.cat([point_flow_features, sampled_context_features], 0)
        rel_pos_point_flow = torch.cat([rel_initial_point_flow, sampled_rel_context_pos], 1)
        features_point_flow = self.self_attn_point_flow(
            query=features_point_flow,
            query_pos=rel_pos_point_flow,
            diff_ts=time_embs,
            context=lang_feat,
            context_pos=None
        )[-1]
        position_point_flow = self.predict_pos_point_flow(
            features_point_flow, rel_pos_point_flow, time_embs, 0, point_flow_features.shape[0], lang_feat
        )
        
        continuous_start = 0
        keyframe_start = self.horizon_continuous
        context_start = self.horizon_continuous + self.horizon_keyframe
        features_keyframe = torch.cat([gripper_features_left[keyframe_start:context_start,:,:], gripper_features_right[keyframe_start:context_start,:,:], features_point_flow], 0)
        rel_pos_keyframe = torch.cat([rel_gripper_pos_left[:,keyframe_start:context_start,:,:], rel_gripper_pos_right[:,keyframe_start:context_start,:,:], rel_pos_point_flow], 1)
        features_keyframe = self.self_attn_keyframe(
            query=features_keyframe,
            query_pos=rel_pos_keyframe,
            diff_ts=time_embs,
            context=lang_feat,
            context_pos=None
        )[-1]

        num_gripper_left_kf = self.horizon_keyframe
        num_gripper_right_kf = self.horizon_keyframe

        # Rotation head
        rotation_left_keyframe = self.predict_rot_left_kf(
            features_keyframe, rel_pos_keyframe, time_embs, 0, num_gripper_left_kf, lang_feat
        )
        rotation_right_keyframe = self.predict_rot_right_kf(
            features_keyframe, rel_pos_keyframe, time_embs, num_gripper_left_kf, num_gripper_left_kf+num_gripper_right_kf, lang_feat
        )

        # Position head
        position_left_keyframe, position_features_left_keyframe = self.predict_pos_left_kf(
            features_keyframe, rel_pos_keyframe, time_embs, 0, num_gripper_left_kf, lang_feat
        )
        position_right_keyframe, position_features_right_keyframe = self.predict_pos_right_kf(
            features_keyframe, rel_pos_keyframe, time_embs, num_gripper_left_kf, num_gripper_left_kf+num_gripper_right_kf, lang_feat
        )

        # Openess head from position head
        openess_left_keyframe = self.openess_predictor_left_kf(position_features_left_keyframe)
        openess_right_keyframe = self.openess_predictor_right_kf(position_features_right_keyframe)
        
        features_keyframe_detach = features_keyframe.detach()
        rel_pos_keyframe_detach = rel_pos_keyframe.detach()
        
        
        
        features_point_flow_detach = features_point_flow.detach()
        rel_pos_point_flow_detach = rel_pos_point_flow.detach()
        
        features_continuous = torch.cat([gripper_features_left[continuous_start:keyframe_start,:,:], gripper_features_right[continuous_start:keyframe_start,:,:],features_keyframe_detach], 0)
        rel_pos_continuous = torch.cat([rel_gripper_pos_left[:,continuous_start:keyframe_start,:,:], rel_gripper_pos_right[:,continuous_start:keyframe_start,:,:], rel_pos_keyframe_detach], 1)
        features_continuous = self.self_attn_continuous(
            query=features_continuous,
            query_pos=rel_pos_continuous,
            diff_ts=time_embs,
            context=lang_feat,
            context_pos=None
        )[-1]
        num_gripper_left_continuous = self.horizon_continuous
        num_gripper_right_continuous = self.horizon_continuous
        
        rotation_left_continuous = self.predict_rot_left_cn(
            features_continuous, rel_pos_continuous, time_embs, 0, num_gripper_left_continuous, lang_feat
        )
        rotation_right_continuous = self.predict_rot_right_cn(
            features_continuous, rel_pos_continuous, time_embs, num_gripper_left_continuous, num_gripper_left_continuous + num_gripper_right_continuous, lang_feat
        )

        # Position head
        position_left_continuous, position_features_left_continuous = self.predict_pos_left_cn(
            features_continuous, rel_pos_continuous, time_embs, 0, num_gripper_left_continuous, lang_feat
        )
        position_right_continuous, position_features_right_continuous = self.predict_pos_right_cn(
            features_continuous, rel_pos_continuous, time_embs, num_gripper_left_continuous, num_gripper_left_continuous + num_gripper_right_continuous, lang_feat
        )

        # Openess head from position head
        openess_left_continuous = self.openess_predictor_left_cn(position_features_left_continuous)
        openess_right_continuous = self.openess_predictor_right_cn(position_features_right_continuous)
        
        position_left = torch.cat((position_left_continuous, position_left_keyframe), 1)
        position_right = torch.cat((position_right_continuous, position_right_keyframe), 1)
        rotation_left = torch.cat((rotation_left_continuous, rotation_left_keyframe), 1)
        rotation_right = torch.cat((rotation_right_continuous, rotation_right_keyframe), 1)
        openess_left = torch.cat((openess_left_continuous, openess_left_keyframe), 1)
        openess_right = torch.cat((openess_right_continuous, openess_right_keyframe), 1)
        return position_left, rotation_left, openess_left, position_right, rotation_right, openess_right, position_point_flow

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

    def predict_pos_point_flow(self, features, rel_pos, time_embs, start_id, end_id,
                    instr_feats):
        # set_trace()
        position_features = self.position_self_attn_point_flow(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]
        position_features = einops.rearrange(
            position_features[start_id:end_id], "npts b c -> b npts c"
        )
        position_features = self.position_proj_point_flow(position_features)  # (B, N, C)
        position = self.position_predictor_point_flow(position_features)
        B, N, _ = position.shape
        position = position.reshape(B, N, 4, 3)
        position = einops.rearrange(
            position, "b n t x -> b t n x"
        )
        position = position.reshape(B,-1,3)
        return position
    
    def predict_pos_left_kf(self, features, rel_pos, time_embs, start_id, end_id,
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
    def predict_pos_left_cn(self, features, rel_pos, time_embs, start_id, end_id,
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
    
    def predict_pos_right_kf(self, features, rel_pos, time_embs, start_id, end_id,
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
    def predict_pos_right_cn(self, features, rel_pos, time_embs, start_id, end_id,
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

    def predict_rot_left_kf(self, features, rel_pos, time_embs, start_id, end_id,
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
    
    def predict_rot_left_cn(self, features, rel_pos, time_embs, start_id, end_id,
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
    
    def predict_rot_right_kf(self, features, rel_pos, time_embs, start_id, end_id,
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
    
    def predict_rot_right_cn(self, features, rel_pos, time_embs, start_id, end_id,
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