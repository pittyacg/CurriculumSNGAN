# Set
CURRICULUM = True
CURRICULUM_MODE = # 'batching' / 'weighting' / 'sampling'

# Lines 605-606 gan_cifar_resnet.py
for i in range(N_CRITIC):
    if CURRICULUM:
        _data, _labels, _scores = next_batch(a_img, a_lab, a_score, BATCH_SIZE, iteration, CURRICULUM_MODE)
    else:
        _data, _labels, _scores = next(gen)


# New function
# Example of parameters for scoring function:
# K_PARAM = 5 * math.pow(10,-5)
# T_PARAM = 4
def next_batch(images_, labels_, scores_, BATCH_SIZE, iteration, mode):
    if mode == 'batching':
        if iteration < CURR_MEDIUM_THRESHOLD:
            th = 1/3
        elif iteration < CURR_HARD_THRESHOLD:
            th = 2/3
        else:
            th = 1
        easy_value = np.quantile(scores_, th)
        idx = np.where(scores <= easy_value)
    elif mode == 'weighting':
        scores_ = 1 + math.exp(-(iteration+1) * K_PARAM) * (-1 * scores_) * T_PARAM
        idx = np.random.choice(len(scores_), BATCH_SIZE)
    elif mode == 'sampling':
        scores_ = 1 + math.exp(-(iteration+1) * K_PARAM) * (-1 * scores_) * T_PARAM
        scores_ = ((scores_-min(scores_))/(max(scores_)-min(scores_))) / np.sum(scores_)
        idx = np.random.choice(len(scores_), BATCH_SIZE, p=scores_)
    else:
        raise('No Curriculum Mode has been selected!')

    final_img = images_[idx]
    final_lab = labels_[idx]
    final_scores = scores_[idx]

    return final_img, final_lab, final_scores

# Add weigthing term to the loss, for example, at line 381 for Hinge loss
if CURRICULUM == True and CURRICULUM_MODE == 'weighting':
    disc_costs.append(disc_real_l + disc_fake_l + tf.reduce_mean(curr_scores))

# Add scores to the model
# line 320
curr_scores = tf.placeholder(tf.float32, shape=[BATCH_SIZE,])
# line 610
_disc_cost, _disc_wgan, _gen_cost, _, summaries = session.run(
      [disc_cost, disc_wgan, gen_cost, disc_train_op, summaries_op],
      feed_dict={all_real_data_int: _data,
      all_real_labels: _labels,
      _iteration: iteration,
      curr_scores: _scores})

# read images, labels and scores
# insert around line 560
a_img, a_lab, a_score = load_simple(['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'], DATA_DIR)

# New read function
# diff.txt contains difficulty scores obtained using the Difficulty Estimator

def load_simple(filenames, data_dir):
    all_data = []
    all_labels = []
    for filename in filenames:
        data, labels = unpickle(os.path.join(data_dir, filename))
        all_data.append(data)
        all_labels.append(labels)
    text_file = open(os.path.join(data_dir, "diff.txt"), "r")
    sc = text_file.read().splitlines()
    text_file.close()
    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    scores = np.array(sc, dtype='f')
    return images,labels,scores



