# Training Details

PPI was tested and evaluated on seven tasks in RLBench2, and the ablation study results are shown in the table below:


![ablation](https://github.com/user-attachments/assets/9a2992e5-e9a9-4346-a0b4-1f221ff176f8)


To accurately reproduce the results in the paper, please strictly adhere to the training hyperparameter settings in the scripts (e.g., batch size, training epochs, lr, etc.). If your server does not support 8-GPU parallel training or requires hyperparameter modifications for other reasons, the task success rate may vary slightly. Generally, we recommend training with more steps and epochs (note that increasing the batch size may reduce the number of steps under the same epoch count, potentially lowering the model success rate).

The ckpts for each task in our paper are shown below:
| task | ckpts (epoch num) |
|:---------:|:---------:|
| ball  | 400/450/500  |
| box  | 250/300/350  |
| drawer  | 350/400/450  |
| dustpan  | 300/350/400  |
| handover_easy  | 200/250/450  |
| laptop  | 350/400/450  |
| tray  | 350/400/450  |
