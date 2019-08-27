import tensorflow as tf
class Net():
    def compute_loss(self, binary_label, score_final, name):
        """
        计算模型的IOU损失
        :param binary_label:每个像素点的标签(B, H, W, C)其中C=1
        :param score_final:每个像素点的输出(B, H, W, 2)，在softmax之前
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # 计算IOU损失
            '''
            print("score_final.get_shape().as_list()", score_final.get_shape().as_list())
            XV = tf.reshape(
                score_final,
                shape=[score_final.get_shape().as_list()[0],
                       score_final.get_shape().as_list()[1] * score_final.get_shape().as_list()[2],
                       score_final.get_shape().as_list()[3]])#decode_logits_reshape的shape也应当为(B, H*W, 2)
            Batch_size,nums_pixels,chanel=XV.get_shape().as_list()
            XV=tf.slice(XV,[0,0,1],[Batch_size,nums_pixels,1])#(B,H*W,1)
            XV = tf.squeeze(XV, axis=[2])#(B,H*W)
            YV = tf.squeeze(binary_label, axis=[3])#binary_label的shape为(B, H, W, C)其中C=1，这里也即丢弃C维度
            YV = tf.reshape(
                YV,
                shape=[YV.get_shape().as_list()[0],
                       YV.get_shape().as_list()[1] * YV.get_shape().as_list()[2]])#(B, H*W)
            YV=tf.cast(YV,tf.float32)
            I_X=tf.multiply(XV,YV)
            I_X_sum=tf.reduce_sum(I_X)
            XV_sum=tf.reduce_sum(XV)
            YV_sum=tf.reduce_sum(YV)
            U_X_sum=tf.subtract(tf.add(XV_sum,YV_sum),I_X_sum)
            IOU=tf.divide(I_X_sum,U_X_sum)
            L_iou=tf.clip_by_value(1-IOU, 0., 1-IOU)
            '''
            binary_soft = tf.nn.softmax(logits=score_final, name='binary_soft')
            XV = tf.reshape(
                binary_soft,
                shape=[binary_soft.get_shape().as_list()[0],
                       binary_soft.get_shape().as_list()[1] * binary_soft.get_shape().as_list()[2],
                       binary_soft.get_shape().as_list()[3]])#(B, H*W, 2)       
            Batch_size,nums_pixels,chanel=XV.get_shape().as_list()
            YV = tf.squeeze(binary_label, axis=[3])#binary_label的shape为(B, H, W, C)其中C=1，这里也即丢弃C维度
            YV = tf.reshape(
                YV,
                shape=[YV.get_shape().as_list()[0],
                       YV.get_shape().as_list()[1] * YV.get_shape().as_list()[2]])#(B, H*W)
            YV_P=tf.cast(YV,tf.float32)
            YV_N=1-YV_P
            #计算正例iou
            XV_P=tf.slice(XV,[0,0,1],[Batch_size,nums_pixels,1])#(B,H*W,1)
            XV_P = tf.squeeze(XV_P, axis=[2])#(B,H*W)
            I_X_P=tf.multiply(XV_P,YV_P)
            I_X_sum_P=tf.reduce_sum(I_X_P)
            XV_P_sum=tf.reduce_sum(XV_P)
            YV_P_sum=tf.reduce_sum(YV_P)
            U_X_sum_P=tf.subtract(tf.add(XV_P_sum,YV_P_sum),I_X_sum_P)
            IOU_P=tf.divide(I_X_sum_P,U_X_sum_P)
            #计算反例iou
            XV_N=tf.slice(XV,[0,0,0],[Batch_size,nums_pixels,1])#(B,H*W,0)
            XV_N = tf.squeeze(XV_N, axis=[2])#(B,H*W)
            I_X_N=tf.multiply(XV_N,YV_N)
            I_X_sum_N=tf.reduce_sum(I_X_N)
            XV_N_sum=tf.reduce_sum(XV_N)
            YV_N_sum=tf.reduce_sum(YV_N)
            U_X_sum_N=tf.subtract(tf.add(XV_N_sum,YV_N_sum),I_X_sum_N)
            IOU_N=tf.divide(I_X_sum_N,U_X_sum_N)
            IOU=tf.divide(IOU_N+IOU_P,2)
            L_iou=tf.clip_by_value(1-IOU, 0., 1-IOU)
            return L_iou
                      
            
