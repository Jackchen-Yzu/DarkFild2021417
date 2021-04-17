from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
import tensorflow.keras.backend as K



#定义恒等模块
def identity_block(X, f, filters, stage, block):
    """
    实现恒等块(包括了网络的主道和shortcut部分 构成恒等模块)
    参数：
        X - 输入的tensor类型的数据，维度为( m, n_H_prev, n_W_prev, n_H_prev )
        f - 整数，指定主路径中间的CONV窗口的维度
        filters - 整数列表，定义了主路径每层的卷积层的过滤器数量
        stage - 整数，根据每层的位置来命名每一层，与block参数一起使用。
        block - 字符串，据每层的位置来命名每一层，与stage参数一起使用。
        
    返回：
        X - 恒等块的输出，tensor类型，维度为(n_H, n_W, n_C)
    
    """
    
    #定义命名规则
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"
    
    #获取过滤器
    F1, F2, F3 = filters              #每一个主通道有三个过滤器
    
    #保存输入数据，将会用于为主路径添加捷径
    X_shortcut = X
    
    #主路径的第一部分
    ##卷积层
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1) ,padding="valid",
               name=conv_name_base+"2a", kernel_initializer=glorot_uniform(seed=0))(X)
    ##归一化
    X = BatchNormalization(axis=3,name=bn_name_base+"2a")(X)
    ##使用ReLU激活函数
    X = Activation("relu")(X)
    
    #主路径的第二部分
    ##卷积层
    X = Conv2D(filters=F2, kernel_size=(f,f),strides=(1,1), padding="same",
               name=conv_name_base+"2b", kernel_initializer=glorot_uniform(seed=0))(X)
    ##归一化
    X = BatchNormalization(axis=3,name=bn_name_base+"2b")(X)
    ##使用ReLU激活函数
    X = Activation("relu")(X)
    
    
    #主路径的第三部分
    ##卷积层
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding="valid",
               name=conv_name_base+"2c", kernel_initializer=glorot_uniform(seed=0))(X)
    ##归一化
    X = BatchNormalization(axis=3,name=bn_name_base+"2c")(X)
    ##没有ReLU激活函数
    

    #最后一步：
    ##将捷径与输入加在一起
    X = Add()([X,X_shortcut])
    ##使用ReLU激活函数
    X = Activation("relu")(X)
    
    return X

###########################################################################################################################
#定义卷积模块
def convolutional_block(X, f, filters, stage, block, s=2):
    """
    实现卷积块（网络主道和带卷积的旁道）
    
    参数：
        X - 输入的tensor类型的变量，维度为( m, n_H_prev, n_W_prev, n_C_prev)
        f - 整数，指定主路径中间的CONV窗口的维度
        filters - 整数列表，定义了主路径每层的卷积层的过滤器数量
        stage - 整数，根据每层的位置来命名每一层，与block参数一起使用。
        block - 字符串，据每层的位置来命名每一层，与stage参数一起使用。
        s - 整数，指定要使用的步幅
    
    返回：
        X - 卷积块的输出，tensor类型，维度为(n_H, n_W, n_C)
    """
    
    #定义命名规则
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base   = "bn"  + str(stage) + block + "_branch"
    
    #获取过滤器数量
    F1, F2, F3 = filters
    
    #保存输入数据
    X_shortcut = X
    
    #主路径
    ##主路径第一部分
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(s,s), padding="valid",
               name=conv_name_base+"2a", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2a")(X)
    X = Activation("relu")(X)
    
    ##主路径第二部分
    X = Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding="same",
               name=conv_name_base+"2b", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2b")(X)
    X = Activation("relu")(X)
    
    ##主路径第三部分
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding="valid",
               name=conv_name_base+"2c", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2c")(X)
    
    #捷径
    X_shortcut = Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s), padding="valid",
               name=conv_name_base+"1", kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3,name=bn_name_base+"1")(X_shortcut)
    
    #最后一步
    X = Add()([X,X_shortcut])
    X = Activation("relu")(X)
    
    return X

###############################################################################################################################
#定义Model类
class RES:
    @staticmethod
    def Build(input_shape = (66, 64, 3), classes = 2):
        X_input = Input(input_shape)
        X = ZeroPadding2D((3, 3))(X_input)
        # stage1
        X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name="conv1",
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name="bn_conv1")(X)
        X = Activation("relu")(X)
        X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

        # stage2
        X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block="a", s=1)
        X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="b")
        X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="c")

        # stage3
        X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block="a", s=2)
        X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="b")
        X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="c")
        X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="d")

        # stage4
        X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block="a", s=2)
        X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="b")
        X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="c")
        X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="d")
        X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="e")
        X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="f")

        # stage5
        X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block="a", s=2)
        X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="b")
        X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="c")

        # 均值池化层
        X = AveragePooling2D(pool_size=(2, 2), padding="same")(X)

        # 输出层
        X = Flatten()(X)
        X = Dense(classes, activation="softmax", name="fc" + str(classes),
                  kernel_initializer=glorot_uniform(seed=0))(X)

        # 创建模型
        model = Model(inputs=X_input, outputs=X)

        return model







