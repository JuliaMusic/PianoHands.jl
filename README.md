# PianoHands.jl
(WIP) Predicting hand assignments in piano MIDI using neural networks

# Dataset preparation

Download PIG v1.2 Dataset to `PianoFingeringDataset` and remove duplicate fingering file, approximately 150 fingering files are required.

# 特征设计

输入： midi note number[21-108] | 和上一音符的公制时间差[ms，Float32] | 左手最后演奏位置[21-108][-1表示未分配] | 右手最后演奏位置[21-108][-1表示未分配] | 
label: onehot向量[0,1] [左手，右手]