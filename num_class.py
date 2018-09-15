import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data",one_hot=True)
node_hl1,node_hl2,node_hl3=500,500,500
nclass=10
batch_size=100
x=tf.placeholder('float',[None,784])
y=tf.placeholder('float')

def nnmodel(data):
    hl1={'weights':tf.Variable(tf.random_normal([784,node_hl1])),'biases':tf.Variable(tf.Variable(tf.random_normal([node_hl1])))}
    hl2={'weights':tf.Variable(tf.random_normal([node_hl1,node_hl2])),'biases':tf.Variable(tf.Variable(tf.random_normal([node_hl2])))}
    hl3={'weights':tf.Variable(tf.random_normal([node_hl2,node_hl3])),'biases':tf.Variable(tf.Variable(tf.random_normal([node_hl3])))}
    opl={'weights':tf.Variable(tf.random_normal([node_hl3,nclass])),'biases':tf.Variable(tf.Variable(tf.random_normal([nclass])))}

    l1=tf.add(tf.matmul(data,hl1['weights']),hl1['biases'])
    l1=tf.nn.relu(l1)
    l2=tf.add(tf.matmul(l1,hl2['weights']),hl2['biases'])
    l2=tf.nn.relu(l2)
    l3=tf.add(tf.matmul(l2,hl3['weights']),hl3['biases'])
    l2=tf.nn.relu(l3)
    output=tf.add(tf.matmul(l3,opl['weights']),opl['biases'])
    
    return output
def train(x):
    pred=nnmodel(x)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
    optimizer=tf.train.AdamOptimizer().minimize(cost)
    epoch=10
    with tf.Session( ) as sess:
        sess.run(tf.global_variables_initializer())
        for xl in range(epoch):
            eloss=0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epx,epy=mnist.train.next_batch(batch_size)
                _,c=sess.run([optimizer,cost],feed_dict={x:epx,y:epy})
                eloss+=c
            print("Epoch:{} at eloss: {}".format(xl,eloss))
        correct=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
        accuracy=tf.reduce_mean(tf.cast(correct,'float'))
        print("Accuracy is ",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
if __name__=="__main__":
    train(x)
