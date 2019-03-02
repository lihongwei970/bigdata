from flask import Flask, jsonify, abort, make_response, request
from redis import Redis, RedisError

import tensorflow as tf
from PIL import Image, ImageFilter

import logging
log = logging.getLogger()
log.setLevel('INFO')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
log.addHandler(handler)

from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement


#from cassandra.cluster import Cluster
#from cassandra import ConsistencyLevel


app = Flask(__name__)
redis = Redis(host="redis", db=0, socket_connect_timeout=2, socket_timeout=2)

tasks = []
def createTable():
    KEYSPACE = "mnistspace"
    #cluster = Cluster(contact_points=['127.0.0.1'],port=9042)
    cluster = Cluster(contact_points=['hongwei-cassandra'],port=9042)
    session = cluster.connect()

    log.info("Creating keyspace...")
    try:
        session.execute("""
           CREATE KEYSPACE %s
           WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }
           """ % KEYSPACE)

        log.info("setting keyspace...")
        session.set_keyspace(KEYSPACE)

        log.info("creating table...")
        session.execute("""
            CREATE TABLE mnisttable (
               id int,
               file text,
               prediction int,
               PRIMARY KEY (id)
            )
            """)
       
    except Exception as e:
        log.error("Unable to create keyspace")
        log.error(e)

def insertKeySpace(id,file,prediction):
   #cluster = Cluster(contact_points=['127.0.0.1'],port=9042)
   cluster = Cluster(['hongwei-cassandra'],port=9042)
   session = cluster.connect()

   KEYSPACE = "mnistspace"

   log.info("setting keyspace...")
   session.set_keyspace(KEYSPACE)

   log.info("inserting ...")
   session.execute("""
       INSERT INTO mnisttable (id,file,prediction)
       values(%(id)s,%(file)s,%(prediction)s)
       """,{'id':id,'file':file,'prediction':prediction})

def predictint(imvalue):
    """
    This function returns the predicted integer.
    The imput is the pixel values from the imageprepare() function.
    """
    
    # Define the model (same as when creating the model file)
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)
    
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)
       
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   
    
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "model2.ckpt")
        #print ("Model restored.")
       
        prediction=tf.argmax(y_conv,1)
        return prediction.eval(feed_dict={x: [imvalue],keep_prob: 1.0}, session=sess)


def imageprepare(argv):
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255)) #creates white canvas of 28x28 pixels
    
    if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0/width*height),0)) #resize height according to ratio width
        if (nheigth == 0): #rare case but minimum is 1 pixel
            nheigth = 1  
        # resize and sharpen
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (4, wtop)) #paste resized image on white canvas
    else:
        #Height is bigger. Heigth becomes 20 pixels. 
        nwidth = int(round((20.0/height*width),0)) #resize width according to ratio height
        if (nwidth == 0): #rare case but minimum is 1 pixel
            nwidth = 1
         # resize and sharpen
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (wleft, 4)) #paste resized image on white canvas
    
    #newImage.save("sample.png")

    tv = list(newImage.getdata()) #get pixel values
    
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    return tva
    #print(tva)


@app.route('/todo/api/v1.0/tasks', methods=['GET'])
def get_tasks():
    try:
        visits = redis.incr("counter")
    except RedisError:
        visits = "<i>cannot connect to Redis, counter disabled</i>"
    if len(tasks) ==0:
        createTable()
        #tasks.append({'id':1,'file':'file','prediction':1})
        return jsonify('table created')
    else: 
        #cluster = Cluster(['127.0.0.1'],port=9042)
        cluster = Cluster(['hongwei-cassandra'],port=9042)
        session = cluster.connect()

        KEYSPACE = "mnistspace"

        log.info("setting keyspace...")
        session.set_keyspace(KEYSPACE)
        result = []
        rows = session.execute('SELECT * FROM mnisttable')
        for row in rows:
            result.append((row[0],row[1],row[2]))
        return jsonify({'Database': result})

@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = [task for task in tasks if task['id'] == task_id]
    if len(task) == 0:
       abort(404)
    return jsonify({'task': task})

@app.route('/todo/api/v1.0/tasks', methods=['POST'])
def show_prediction():
    if not request.json or not 'file' in request.json:
        abort(400)
    imvalue = imageprepare(request.json['file'])
    predint = predictint(imvalue)
    #cluster = Cluster(['127.0.0.1'],port=9042)
    cluster = Cluster(['hongwei-cassandra'],port=9042)
    session = cluster.connect()

    KEYSPACE = "mnistspace"

    log.info("setting keyspace...")
    session.set_keyspace(KEYSPACE)
    result = []
    rows = session.execute('SELECT * FROM mnisttable')
    for row in rows:
        result.append((row[0]))

    if len(result)==0:
        id_num = 1
    else:
        id_num = len(result)+1

    task = {
        'id': id_num,
        'file': request.json['file'],
        'prediction': int(predint[0])
    }
    tasks.append(task)
    insertKeySpace(id_num,request.json['file'],int(predint[0]))
    return jsonify({'task': task}), 201


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=80,debug = True)