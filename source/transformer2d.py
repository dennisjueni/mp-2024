import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from models import BaseModel
from data import AMASSBatch # type: ignore
from losses import mse  # type: ignore
from configuration import CONSTANTS as C


class ParallelAttention(nn.Module):
    def __init__(
        self,
        num_joints: int,
        d_model: int,
        num_heads_temporal: int,
        num_heads_spacial: int,
        dropout_rate: float,
        dff: int,
    ):
        super(ParallelAttention, self).__init__()
        self.num_joints = num_joints
        self.d_model = d_model
        self.num_heads_temporal = num_heads_temporal
        self.num_heads_spacial = num_heads_spacial
        self.dropout_rate = dropout_rate
        self.dff = dff
        
        self.a = nn.Parameter(torch.rand(1))

        # Define the Q,K,V matrices for each joint in the temporal attention
        self.temp_query_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_model) for _ in range(self.num_joints)]
        )
        self.temp_key_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_model) for _ in range(self.num_joints)]
        )
        self.temp_value_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_model) for _ in range(self.num_joints)]
        )
        self.temp_output_dense = nn.Linear(self.d_model, self.d_model)

        # Define a single K,V matrix, and a Q matrix for each joint in the spacial attention
        self.spac_key_layer = nn.Linear(self.d_model, self.d_model)
        self.spac_value_layer = nn.Linear(self.d_model, self.d_model)
        self.spac_query_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_model) for _ in range(self.num_joints)]
        )
        self.spac_output_dense = nn.Linear(self.d_model, self.d_model)

        self.temp_dropout = nn.Dropout(self.dropout_rate)
        self.temp_layer_norm = nn.LayerNorm(self.d_model)

        self.spac_dropout = nn.Dropout(self.dropout_rate)
        self.spac_layer_norm = nn.LayerNorm(self.d_model)

        self.ff1_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.dff) for _ in range(self.num_joints)]
        )
        self.ff2_layers = nn.ModuleList(
            [nn.Linear(self.dff, self.d_model) for _ in range(self.num_joints)]
        )

        self.ffn_dropout = nn.Dropout(self.dropout_rate)
        self.ffn_layer_norm = nn.LayerNorm(self.d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        The scaled dot product attention mechanism introduced in the Transformer
        :param q: the query vectors matrix (..., attn_dim, d_model/num_heads)
        :param k: the key vector matrix (..., attn_dim, d_model/num_heads)
        :param v: the value vector matrix (..., attn_dim, d_model/num_heads)
        :param mask: a mask for attention
        :return: the updated encoding and the attention weights matrix
        """
        # attn_dim: num_joints for spatial and seq_len for temporal

        matmul_qk = torch.matmul(
            q, k.transpose(-2, -1)
        )  # (..., num_heads, attn_dim, attn_dim)

        # scale matmul_qk
        dk = k.size(-1)
        scaled_attention_logits = matmul_qk / torch.sqrt(
            torch.tensor(dk, dtype=torch.float32).to(device=C.DEVICE)
        )

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += mask * -1e9

        # normalized on the last axis (seq_len_k) so that the scores add up to 1.
        attention_weights = F.softmax(
            scaled_attention_logits, dim=-1
        )  # (..., num_heads, attn_dim, attn_dim)

        output = torch.matmul(attention_weights, v)  # (..., num_heads, attn_dim, depth)

        return output, attention_weights

    def sep_split_heads(self, x, batch_size, seq_len, num_heads):
        """
        split the embedding vector for different heads for the temporal attention
        :param x: the embedding vector (batch_size, seq_len, d_model)
        :param batch_size: batch size
        :param seq_len: sequence length
        :param num_heads: number of temporal heads
        :return: the split vector (batch_size, num_heads, seq_len, depth)
        """
        depth = self.d_model // num_heads
        x = x.view(batch_size, seq_len, num_heads, depth)
        return x.permute(0, 2, 1, 3)

    def sep_temporal_attention(self, x, mask):
        """
        the temporal attention block
        :param x: the input (batch_size, seq_len, num_joints, d_model)
        :param mask: temporal mask (usually the look ahead mask)
        :return: the output (batch_size, seq_len, num_joints, d_model)
        """

        outputs = []
        attn_weights = []

        batch_size = x.size(0)
        seq_len = x.size(1)

        x = x.permute(2, 0, 1, 3)  # (num_joints, batch_size, seq_len, d_model)

        # different joints have different embedding matrices
        for joint_idx in range(self.num_joints):

            # get the representation vector of the joint
            joint_rep = x[joint_idx]  # (batch_size, seq_len, d_model)

            # embed it to query, key and value vectors
            q = self.temp_query_layers[joint_idx](
                joint_rep
            )  # (batch_size, seq_len, d_model)
            k = self.temp_key_layers[joint_idx](
                joint_rep
            )  # (batch_size, seq_len, d_model)
            v = self.temp_value_layers[joint_idx](
                joint_rep
            )  # (batch_size, seq_len, d_model)

            # split it to several attention heads
            q = self.sep_split_heads(q, batch_size, seq_len, self.num_heads_temporal)
            # (batch_size, num_heads, seq_len, depth)
            k = self.sep_split_heads(k, batch_size, seq_len, self.num_heads_temporal)
            # (batch_size, num_heads, seq_len, depth)
            v = self.sep_split_heads(v, batch_size, seq_len, self.num_heads_temporal)
            # (batch_size, num_heads, seq_len, depth)
            # calculate the updated encoding by scaled dot product attention
            scaled_attention, attention_weights = self.scaled_dot_product_attention(
                q, k, v, mask
            )
            # (batch_size, num_heads, seq_len, depth)
            scaled_attention = scaled_attention.permute(0, 2, 1, 3)
            # (batch_size, seq_len, num_heads, depth)

            # concatenate the outputs from different heads
            concat_attention = scaled_attention.reshape(
                batch_size, seq_len, self.d_model
            )
            # (batch_size, seq_len, d_model)

            # go through a fully connected layer
            output = self.temp_output_dense(concat_attention).unsqueeze(2)
            # (batch_size, seq_len, 1, d_model)
            outputs.append(output)

            last_attention_weights = attention_weights[
                :, :, -1, :
            ]  # (batch_size, num_heads, seq_len)
            attn_weights.append(last_attention_weights)

        return (
            torch.cat(outputs, dim=2).to(device=C.DEVICE),
            torch.stack(attn_weights, dim=0).to(device=C.DEVICE),
        )

    def split_heads(self, x, shape0, shape1, attn_dim, num_heads):
        """
        split the embedding vector for different heads for the spatial attention
        :param x: the embedding vector (batch_size, seq_len, num_joints, d_model)
        :param shape0: batch size
        :param shape1: sequence length
        :param attn_dim: number of joints
        :param num_heads: number of heads
        :return: the split vector (batch_size, seq_len, num_heads, num_joints, depth)
        """
        depth = self.d_model // num_heads
        x = x.view(shape0, shape1, attn_dim, num_heads, depth)
        return x.permute(0, 1, 3, 2, 4)

    def sep_spacial_attention(self, x, mask):
        """
        the spatial attention block
        :param x: the input (batch_size, seq_len, num_joints, d_model)
        :param mask: spatial mask (usually None)
        :return: the output (batch_size, seq_len, num_joints, d_model)
        """
        # embed each vector to key, value and query vectors
        k = self.spac_key_layer(x)  # (batch_size, seq_len, num_joints, d_model)
        v = self.spac_value_layer(x)  # (batch_size, seq_len, num_joints, d_model)

        # Different joints have different query embedding matrices
        x = x.permute(2, 0, 1, 3)  # (num_joints, batch_size, seq_len, d_model)
        q_joints = []
        for joint_idx in range(self.num_joints):
            q = self.spac_query_layers[joint_idx](x[joint_idx]).unsqueeze(
                2
            )  # (batch_size, seq_len, d_model)
            q_joints += [q]

        q_joints = torch.cat(
            q_joints, dim=2
        )  # (batch_size, seq_len, num_joints, d_model)
        batch_size = q_joints.size(0)
        seq_len = q_joints.size(1)

        # split it to several attention heads
        q_joints = self.split_heads(
            q_joints, batch_size, seq_len, self.num_joints, self.num_heads_spacial
        )
        # (batch_size, seq_len, num_heads, num_joints, depth)
        k = self.split_heads(
            k, batch_size, seq_len, self.num_joints, self.num_heads_spacial
        )
        # (batch_size, seq_len, num_heads, num_joints, depth)
        v = self.split_heads(
            v, batch_size, seq_len, self.num_joints, self.num_heads_spacial
        )
        # (batch_size, seq_len, num_heads, num_joints, depth)

        # calculate the updated encoding by scaled dot product attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q_joints, k, v, mask
        )
        # (batch_size, seq_len, num_heads, num_joints, depth)
        # concatenate the outputs from different heads
        scaled_attention = scaled_attention.permute(0, 1, 3, 2, 4)
        concat_attention = scaled_attention.reshape(
            batch_size, seq_len, self.num_joints, self.d_model
        )
        # (batch_size, seq_len, num_joints, d_model)

        # go through a fully connected layer
        output = self.spac_output_dense(concat_attention)

        attention_weights = attention_weights[
            :, -1, :, :, :
        ]  # (batch_size, num_heads, num_joints, num_joints)

        return output, attention_weights

    def point_wise_feed_forward_network(self, inputs):
        """
        The feed forward block
        :param inputs: inputs (batch_size, seq_len, num_joints, d_model)
        :return: outputs (batch_size, seq_len, num_joints, d_model)
        """
        inputs = inputs.permute(
            2, 0, 1, 3
        )  # (num_joints, batch_size, seq_len, d_model)
        outputs = []
        # different joints have different embedding matrices
        for idx in range(self.num_joints):
            joint_inputs = inputs[idx]  # (batch_size, seq_len, d_model)

            joint_outputs = F.relu(
                self.ff1_layers[idx](joint_inputs)
            )  # (batch_size, seq_len, dff)
            joint_outputs = self.ff2_layers[idx](
                joint_outputs
            )  # (batch_size, seq_len, d_model)
            outputs += [joint_outputs]

        outputs = torch.cat(
            outputs, dim=-1
        )  # (batch_size, seq_len, num_joints * d_model)
        outputs = outputs.reshape(
            outputs.size(0), outputs.size(1), self.num_joints, self.d_model
        )
        return outputs

    def forward(self, x, look_ahead_mask):
        """
        The layer with spatial and temporal blocks in parallel
        :param x: the input (batch_size, seq_len, num_joints, d_model)
        :param look_ahead_mask: the look ahead mask
        :return: outputs (batch_size, seq_len, num_joints, d_model) and the attention blocks
        """
        # temporal attention
        attn1, attn_weights_block1 = self.sep_temporal_attention(x, look_ahead_mask)

        if self.training:
            attn1 = self.temp_dropout(attn1)

        temporal_out = self.temp_layer_norm(attn1 + x)

        # spatial attention
        attn2, attn_weights_block2 = self.sep_spacial_attention(x, None)

        if self.training:
            attn2 = self.spac_dropout(attn2)

        spatial_out = self.spac_layer_norm(attn2 + x)

        # Pass 'a' through its layer
        # Apply sigmoid to 'a' to ensure it's between 0 and 1
        a = torch.sigmoid(self.a)
        
        # add the temporal output and the spatial output
        out = a*temporal_out + (1 - a)*spatial_out

        # feed forward
        ffn_output = self.point_wise_feed_forward_network(out)

        if self.training:
            ffn_output = self.ffn_dropout(ffn_output)

        final = self.ffn_layer_norm(ffn_output + out)

        return final, attn_weights_block1, attn_weights_block2


transformer_config = {
    "transformer_d_model": 128,
    "transformer_dff": 256,
    "transformer_dropout_rate": 0.15,
    "transformer_lr": 0.00005,
    "transformer_num_heads_spacial": 8,
    "transformer_num_heads_temporal": 8,
    "transformer_num_layers": 8,
    "transformer_warm_up_steps": 8000,
    "transformer_window_length": 120,
    "transformer_num_joints": 15,
    "transformer_joint_size": 9,
}


class Transformer(BaseModel):
    """Our implementation of the Transformer model from the Motion Transformer paper."""

    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.pose_size = config.pose_size

        self.d_model: int = transformer_config["transformer_d_model"]
        self.dff: int = transformer_config["transformer_dff"]
        self.dropout_rate: int = transformer_config["transformer_dropout_rate"]
        self.lr: int = transformer_config["transformer_lr"]
        self.num_heads_spacial: int = transformer_config[
            "transformer_num_heads_spacial"
        ]
        self.num_heads_temporal: int = transformer_config[
            "transformer_num_heads_temporal"
        ]
        self.num_layers: int = transformer_config["transformer_num_layers"]
        self.warm_up_steps: int = transformer_config["transformer_warm_up_steps"]
        self.window_len: int = transformer_config["transformer_window_length"]
        self.num_joints: int = transformer_config["transformer_num_joints"]
        self.joint_size: int = transformer_config["transformer_joint_size"]

        self.pos_encoding = self.positional_encoding()
        self.look_ahead_mask = self.create_look_ahead_mask()

        self.create_model()

    def create_model(self):

        self.input_dropout = nn.Dropout(self.dropout_rate)

        self.embedding_layers = nn.ModuleList(
            [nn.Linear(self.joint_size, self.d_model) for _ in range(self.num_joints)]
        )

        self.para_attention_layers = nn.ModuleList(
            [
                ParallelAttention(
                    self.num_joints,
                    self.d_model,
                    self.num_heads_temporal,
                    self.num_heads_spacial,
                    self.dropout_rate,
                    self.dff,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.decoding_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.joint_size) for _ in range(self.num_joints)]
        )

    def model_name(self):
        """A summary string of this model. Override this if desired."""
        return "torch_transformer"

    def learning_rate_scheduler(self, global_step):
        d_model = torch.tensor(self.d_model).float()
        arg1 = torch.rsqrt(torch.tensor(global_step))
        arg2 = torch.tensor(global_step * (self.warm_up_steps**-1.5))
        ret = torch.rsqrt(d_model) * torch.min(arg1, arg2)

        return ret.to(device=C.DEVICE)
    
    def finetuning_lr_scheduler(self, global_step):
        factor = 0.5 ** (global_step // 1000)

        # Update the current learning rate
        current_lr = self.lr * factor

        # Convert the learning rate to a tensor and move it to the specified device
        ret = torch.tensor(current_lr).to(device=C.DEVICE)

        return ret

    def create_look_ahead_mask(self):
        """
        Create a look ahead mask given a certain window length.
        :return: the mask (window_length, window_length)
        """
        size = self.window_len

        mask = torch.ones((size, size)).triu(
            diagonal=1
        )  # Generate an upper triangular matrix
        return mask.to(device=C.DEVICE)  # (seq_len, seq_len)

    def get_angles(self, pos, i):
        """
        calculate the angles givin postion and i for the positional encoding formula
        :param pos: pos in the formula
        :param i: i in the formula
        :return: angle rad
        """
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(self.d_model))
        return pos * angle_rates

    def positional_encoding(self):
        """
        Calculate the positional encoding given the window length.
        :return: positional encoding (1, window_length, 1, d_model)
        """
        angle_rads = self.get_angles(
            np.arange(self.window_len)[:, np.newaxis],
            np.arange(self.d_model)[np.newaxis, :],
        )

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, :, np.newaxis, :]

        ret = torch.tensor(pos_encoding).float()  # (1, seq_len, 1, d_model)

        return ret.to(device=C.DEVICE)

    def transformer(self, inputs, look_ahead_mask=None):
        """
        The attention blocks
        :param inputs: inputs (batch_size, seq_len, num_joints, joint_size)
        :param look_ahead_mask: the look ahead mask
        :return: outputs (batch_size, seq_len, num_joints, joint_size)
        """

        # encode each rotation matrix to the feature space (d_model)
        # different joints have different encoding matrices
        inputs = inputs.permute(
            2, 0, 1, 3
        )  # (num_joints, batch_size, seq_len, joint_size)
        embed = []
        for joint_idx in range(self.num_joints):
            joint_rep = self.embedding_layers[joint_idx](
                inputs[joint_idx]
            )  # (batch_size, seq_len, d_model)
            embed.append(joint_rep)

        x = torch.cat(embed, dim=-1)  # (batch_size, seq_len, num_joints * d_model)
        x = x.view(
            x.size(0), x.size(1), self.num_joints, self.d_model
        )  # (batch_size, seq_len, num_joints, d_model)

        # add the positional encoding
        x += self.pos_encoding

        if self.training:
            x = self.input_dropout(x)

        # put into several attention layers
        # (batch_size, seq_len, num_joints, d_model)
        attention_weights_temporal = []
        attention_weights_spatial = []
        attention_weights = {}

        for i in range(self.num_layers):
            x, block1, block2 = self.para_attention_layers[i](x, look_ahead_mask)
            attention_weights_temporal += [
                block1
            ]  # (batch_size, num_joints, num_heads, seq_len)
            attention_weights_spatial += [
                block2
            ]  # (batch_size, num_heads, num_joints, num_joints)
        # (batch_size, seq_len, num_joints, d_model)

        attention_weights["temporal"] = torch.stack(
            attention_weights_temporal, dim=1
        )  # (batch_size, num_layers, num_joints, num_heads, seq_len)
        attention_weights["spatial"] = torch.stack(
            attention_weights_spatial, dim=1
        )  # (batch_size, num_layers, num_heads, num_joints, num_joints)

        # decode each feature to the rotation matrix space
        # different joints have different decoding matrices
        x = x.permute(2, 0, 1, 3)  # (num_joints, batch_size, seq_len, joint_size)
        output = []
        for joint_idx in range(self.num_joints):
            joint_inputs = x[joint_idx]  # Access input for the current joint
            # Pass through the respective decoding layer
            joint_output = self.decoding_layers[joint_idx](joint_inputs)
            output.append(joint_output.unsqueeze(0))

        final_output = torch.cat(output, dim=0)  # Concatenate along the new dimension
        final_output = final_output.permute(
            1, 2, 0, 3
        )  # Reorder back to (batch_size, seq_len, num_joints, joint_size)

        return final_output, attention_weights

    def forward(self, batch: AMASSBatch):
        """The forward pass."""

        # batch.poses.shape = ([16, 144, 135])

        batch_size = batch.batch_size
        model_out = {
            "seed": batch.poses[:, : self.config.seed_seq_len],
            "predictions": None,
        }

        if self.training:

            # Sequence from 0 to 120
            target_input = batch.poses[:, : self.window_len, :]
            sequence_length = target_input.shape[1]

            assert sequence_length == 120

            outputs, _ = self.transformer(
                target_input.view(
                    batch_size, sequence_length, self.num_joints, self.joint_size
                ),
                self.look_ahead_mask,
            )

            outputs = outputs.reshape(batch_size, sequence_length, -1)  # (16, 120, 135)

            # The residual velocity stuff
            outputs += target_input

            model_out["predictions"] = outputs

        else:
            # In the evaluation phase, we start with the seed sequence and predict the future target_seq_len poses autoregressively.
            inputs = batch.poses[:, : self.config.seed_seq_len, :]
            num_steps = self.config.target_seq_len
            predictions = []

            for _ in range(num_steps):

                outputs, _ = self.transformer(
                    inputs.view(
                        batch_size, self.window_len, self.num_joints, self.joint_size
                    ),
                    self.look_ahead_mask,
                )

                outputs = outputs.reshape(
                    batch_size, self.window_len, -1
                )  # (16, 120, 135)

                # The residual velocity stuff
                outputs += inputs

                # Append the last prediction to the list
                predictions.append(outputs[:, -1:, :])

                # Concatenate the last prediction to the input sequence
                inputs = torch.cat((inputs, predictions[-1]), dim=1)
                inputs = inputs[:, -self.window_len :, :]

            model_out["predictions"] = torch.cat(predictions, dim=1)

        return model_out

    def backward(self, batch: AMASSBatch, model_out):
        """The backward pass."""
        predictions_pose = model_out["predictions"]  # (16, 120, 135)

        if self.training:
            # Sequence from 1 to 121
            targets_pose = batch.poses[:, 1 : self.window_len + 1, :]
            seq_len = targets_pose.shape[1]

            assert seq_len == 120

            diff = targets_pose - predictions_pose
            per_joint_loss = torch.square(diff).view(
                -1, seq_len, self.num_joints, self.joint_size
            )
            per_joint_loss = torch.sqrt(torch.sum(per_joint_loss, dim=-1))
            per_joint_loss = torch.sum(per_joint_loss, dim=-1)
            loss_ = torch.mean(per_joint_loss)

            print(f"Dennis Loss: {loss_}")

            loss_vals = {"total_loss": loss_.cpu().item()}
            loss_.backward()

        else:
            targets_pose = batch.poses[:, -self.config.target_seq_len :, :]

            total_loss = mse(predictions_pose, targets_pose)
            loss_vals = {"total_loss": total_loss.cpu().item()}

        return loss_vals, targets_pose
