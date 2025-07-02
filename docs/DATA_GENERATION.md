# Data Generation
Here we provide the guide for regenerating the RLBench2 dataset with object 6d pose metadata.

## Environment
Please refer to [INSTALLATION.md](INSTALLATION.md) for environment configuration. (Please use [my RLBench2 repo](https://github.com/yuyinyang3y/RLBench).)

## Specify the task name and object name
### 1. Task Name
The script to generate the data is `PPI/scripts/data_generation/generator_meta_data.sh`. Change the `tasks` parameter to the task name you want to generate the data for. Also, specify the `save_path` parameter to the path to save the generated data and `episodes_per_task` parameter to the number of episodes you want to generate for each task.

### 2. Object Name
The object name is specified in the line 366 in `RLBench/rlbench/backend/scene_gen.py`. For example, for the `bimanual_lift_ball` task, the object name is `ball`.

The following is the mapping between the task name and the object name:
```yaml
bimanual_handover_item_easy: object_nh = Shape('item')
bimanual_lift_ball: object_nh = Shape('ball')
bimanual_lift_tray: object_nh = Shape('tray')
bimanual_push_box: object_nh = Shape('cube')
bimanual_put_item_in_drawer: object_nh = Shape('item')
bimanual_sweep_to_dustpan: object_nh = Shape('broom')
bimanual_pick_laptop: object_nh = Shape('lid')
```

To get the object name for other tasks, you may refer to the task files under the `RLBench/rlbench/bimanual_tasks` directory.

## Run the script
```bash
bash PPI/scripts/data_generation/generator_meta_data.sh
```

