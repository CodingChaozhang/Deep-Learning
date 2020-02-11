实现方法:
年龄每1岁为1类, 共70类, image_size=224, batch_size=32, 然后训练.
主要用了keras框架,densenet121预训练模型,Adam优化器,label smooth策略,模型融合

主要代码文件说明:
data文件夹下是数据集以及生成数据list.txt文件的相关处理

train_densenet121.py: keras框架在train_densenet121.py文件中进行训练,参数直接修改即可,将测试效果较好的模型放入指定文件夹keras_models中

keras_val_save.py: 将keras_models文件夹中的模型分别输出验证集val结果和测试集test结果,分别放入models_val_outputs文件夹和models_test_outputs文件夹,
结果文件txt以模型的名字命名.

combination_all.py: 进行模型融合:将models_val_outputs文件夹中的所有输出结果全部融合(从一个到全部组合),结果放入all_fusion.txt.

test_to_csv.py: 从all_fusion.txt中选取准确率最高的几个模型,输出测试集test结果到csv文件.


其它python文件说明:
keras_test.py: 刚开始写的输出到csv文件代码, 许多地方可以化简, 和test_to_csv.py功能相同
keras_test_all.py: 和combination_all.py文件功能类似,进行模型融合测试(早期结果放入到了keras_fusion.txt文件夹)
keras_test_tqdm.py: 验证单个模型的准确率,也可以直接将结果输出到csv文件(是单个模型的),分为一次性加载所有数据和分批加载


最好结果的实验步骤:
训练代码:
train_densenet121.py

①首先用densenet121预训练模型直接训练,学习率从1e-3 -> 2e-5 得到准确率为0.2397的baseline模型,保存为densenet121_0.2397_baseline.hdf5
②使用label smooth策略,将α设置为0.7999,其他设置为0.0029 (0.7999+0.0029*69=1),其他参数不变,训练得到准确率为0.2455的模型,保存为densenet121_0.2455_smt0.8.hdf5
③将②中α设置为0.655,其他设置为0.005,训练得到准确率为0.2474的模型,保存为densenet121_0.2474_smt0.655.hdf5
④将②中α设置为0.448,其他设置为0.008,训练得到准确率为0.2569的模型,保存为densenet121_0.2560_smt0.448.hdf5
⑤将上四中得到的4个模型进行融合,得到的即为最终结果,输出到csv文件中即可.

