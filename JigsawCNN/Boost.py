'''
CNN boost training
'''


import numpy as np
import tensorflow as tf
import JIgsawAbitraryNetROI
import Parameters
import os, glob
import TFRecordIOWithROI
import cv2
import sys
import Utils
import PairwiseAlignment2Image
import argparse

Args = []

def BoostTraining(net,
                  input, roi_box, target, weights, data_ids,
                  input_test, roi_box_test, target_test, weights_test, data_ids_test,
                  tensorboard_dir, checkpoint_dir,
                  is_training=True):

    import time

    # TRAINING graph
    target_value = target
    gt_classification = tf.argmax(target, axis=1, name="gt_classification")
    logits = net._inference(input, roi_box, is_training)

    with tf.name_scope('metrics'):
        pred_class = tf.argmax(logits, axis=1, name="predicted_class")
        correct_prediction = tf.equal(pred_class, gt_classification)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        precision_op = tf.metrics.precision(gt_classification, pred_class)
        recall_op = tf.metrics.recall(gt_classification, pred_class)

        # Use running values (precision_op[1], recall_op[1])
        precision = precision_op[1]
        recall = recall_op[1]
        epsilon = 1e-7
        f1_score = 2 * precision * recall / (precision + recall + epsilon)

        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('precision', precision)
        tf.summary.scalar('recall', recall)
        tf.summary.scalar('f1_score', f1_score)

    losses = net._loss(logits, target_value, weights, data_ids)
    global_step = tf.Variable(0, trainable=False, name="global_step")
    opt_op = net._optmization(losses=losses, global_step=global_step)

    # TEST graph
    with tf.name_scope('test'):
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=True):
            logits_test = net._inference(input_test, roi_box_test, is_training=False)
        gt_classification_test = tf.argmax(target_test, axis=1, name="gt_classification_test")
        pred_test = tf.argmax(logits_test, axis=1, name="test_prediction")
        correct_prediction_test = tf.equal(pred_test, gt_classification_test)
        test_accuracy = tf.reduce_mean(tf.cast(correct_prediction_test, tf.float32))
        tf.summary.scalar('test_accuracy', test_accuracy)

    merged = tf.summary.merge_all()

    # Init ops and saver
    sess_init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep=2)

    with tf.Session() as sess:
        print("🚀 Initializing session...")
        sess.run(sess_init_op)
        print("✅ Session initialized.")

        tensorboard_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print("🧪 Testing input pipeline before starting training...")

        try:
            sample_input, sample_target = sess.run([input, target])
            sample_input_test, sample_target_test = sess.run([input_test, target_test])
            print("✅ Successfully fetched training and test batches.")
            print("   Train Input shape:", sample_input.shape)
            print("   Test Input shape:", sample_input_test.shape)
        except Exception as e:
            print("❌ ERROR fetching input data:", e)
            coord.request_stop()
            coord.join(threads)
            return

        print("🏋️ Starting training loop...")
        
        best_test_accuracy = 0.0
        best_test_f1 = 0.0

        while global_step.eval() < Parameters.NNHyperparameters['total_training_step']:
            step = global_step.eval()

            if step % 1000 == 0 and step != 0:
                print("💾 Checkpointing at step", step, end='...')
                saver.save(sess, checkpoint_dir + '/', global_step=global_step)
                print(" Done.")

            print(f"🌀 Step {step} - running training op...")

            try:
                _, la, value_loss, acc, precision_val, recall_val, f1_val, summary = sess.run([
                    opt_op, target, losses['value_loss'], accuracy,
                    precision, recall, f1_score, merged
                ])
                print("✅ Step complete.")
            except Exception as e:
                print("❌ ERROR during training step:", e)
                break

            if step % 10 == 0:
                tensorboard_writer.add_summary(summary, step)

            if step % 100 == 0:
                acc_test, precision_val_test, recall_val_test, f1_test = sess.run(
                    [test_accuracy, precision, recall, f1_score]
                )
                print(f"🧪 Test accuracy at step {step}: {acc_test:.4f}")
                print(f"🧪 Test F1 score at step {step}: {f1_test:.4f}")
                print(f"🧪 Test precision at step {step}: {precision_val_test:.4f}")
                print(f"🧪 Test recall at step {step}: {recall_val_test:.4f}")

                with open(os.path.join(checkpoint_dir, "training_log.txt"), "a") as logf:
                    logf.write(
                        f"Step {step} - Test Accuracy: {acc_test:.4f}, "
                        f"Precision: {precision_val_test:.4f}, "
                        f"Recall: {recall_val_test:.4f}, "
                        f"F1: {f1_test:.4f}\n"
                    )

                if acc_test > best_test_accuracy:
                    best_test_accuracy = acc_test
                    best_path = os.path.join(checkpoint_dir, "best_model_accuracy.ckpt")
                    print(f"🏆 New best **accuracy** model found! Saving to {best_path}...")
                    saver.save(sess, best_path)

                if f1_test > best_test_f1:
                    best_test_f1 = f1_test
                    best_path_f1 = os.path.join(checkpoint_dir, "best_model_f1.ckpt")
                    print(f"🥇 New best **F1** model found! Saving to {best_path_f1}...")
                    saver.save(sess, best_path_f1)

            print(f"📊 value_loss: {value_loss:.4f}")
            print(f"📈 train accuracy: {acc:.4f}")
            print(f"📐 precision: {precision_val:.4f}")
            print(f"🔁 recall: {recall_val:.4f}")
            print(f"🎯 F1 score: {f1_val:.4f}")
            print("---------------------------")

        tensorboard_writer.close()
        print("📁 TensorBoard graph saved to:", tensorboard_dir)
        
        # Recalculate final metrics
        acc_test_final, precision_test_final, recall_test_final, f1_test_final = sess.run(
            [test_accuracy, precision, recall, f1_score]
        )

        # Save final test metrics to a log file
        final_log_path = os.path.join(checkpoint_dir, "final_results.txt")
        with open(final_log_path, "w") as f:
            f.write("Final test results after training:\n")
            f.write(f"Test Accuracy: {acc_test_final:.4f}\n")
            f.write(f"Precision: {precision_test_final:.4f}\n")
            f.write(f"Recall: {recall_test_final:.4f}\n")
            f.write(f"F1 Score: {f1_test_final:.4f}\n")
        print(f"📝 Final test results saved to {final_log_path}")

        print(f"💾 Saving final checkpoint to {checkpoint_dir}...", end='')
        saver.save(sess, checkpoint_dir + "/", global_step=global_step)
        print(" Done!")

        coord.request_stop()
        coord.join(threads)     

'''
Allow restore variables even though some new variables have been added after training
see https://github.com/tensorflow/tensorflow/issues/312
'''
def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = tf.get_variable(saved_var_name)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


def Evaluation(net, input, roi_box, target, weights, data_ids, checkpoint_dir, is_training=False):
    new_weights = np.zeros(weights.shape[0])

    gt_classification = tf.argmax(target, dimension=1, name="gt_classification")
    logits = net._inference(input, roi_box, is_training)
    with tf.name_scope('calculation_accuracy'):
        I = tf.not_equal(net.pred, gt_classification)
        I = tf.cast(I, dtype=tf.float32)
        corresponding_weights = tf.gather(weights, data_ids)
        corresponding_weights = tf.reshape(corresponding_weights, [-1])
        I = tf.reshape(I, [-1])
        batch_err = tf.reduce_sum(tf.multiply(corresponding_weights, I))        # Numerator of Ek
        batch_weight = tf.reduce_sum(corresponding_weights)                     # denominator of Ek

        correct_prediction = tf.equal(net.pred, gt_classification)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        pred_result = tf.cast(correct_prediction, tf.int32)
        minus_yg = tf.add(tf.multiply(pred_result, -2), 1)
        minus_yg = tf.cast(minus_yg, tf.float32)
        new_weight = tf.multiply(corresponding_weights, tf.exp(minus_yg))  # w_{k-1,i}exp(-y_i*G_k(x_i))

    saver = tf.train.Saver(max_to_keep=2)
    sess_init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    count = 0
    w = 0
    wI = 0
    with tf.Session() as sess:
        sess.run(sess_init_op)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir + '/'))
        # optimistic_restore(session=sess, save_file=checkpoint_dir + '/-5')
        print("model restored!")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            while True:
                a, b, pre, acc, pred, gt, im, la, bat_e, bat_w, bat_new_w, ids = sess.run([minus_yg,pred_result, correct_prediction, accuracy, net.pred, gt_classification, input, target, batch_err, batch_weight, new_weight, data_ids])
                wI+=bat_e
                w+=bat_w
                ids = np.reshape(ids, [-1])
                count += len(ids)
                new_weights[ids] = bat_new_w

                progress = "evaluate %d/%d" % (count, len(new_weights))
                sys.stdout.write('\r' + progress)
                for i in range(len(pred)):
                    if pred[i] == gt[i] and pred[i] == 0:
                        tn += 1
                    if pred[i] == gt[i] and pred[i] == 1:
                        tp += 1
                    if pred[i] != gt[i] and pred[i] == 0:
                        fn += 1
                    if pred[i] != gt[i] and pred[i] == 1:
                        fp += 1
        except tf.errors.OutOfRangeError:
            print('\nDone testing -- epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads)

    print("tp, tn, fp, fn: %d, %d, %d, %d" % (tp, tn, fp, fn))
    Ek = wI/w
    alpha = 0.5*np.log((1-Ek)/Ek)
    new_weights = np.exp(alpha)*new_weights
    return alpha, new_weights

'''
Evaluate image directly
'''
def SingleTest(checkpoint_root, K, net, is_training=False):
    input = tf.placeholder(tf.float32, [None, net.params['height'], net.params['width'], net.params['depth']])
    roi_box = tf.placeholder(tf.float32, [None, 4])

    logits = net._inference(input, roi_box, is_training)
    probability = tf.nn.softmax(logits)

    '''restore sessions'''
    sessions = []
    saver = tf.train.Saver(max_to_keep=2)
    for i in range(K):
        check_point = os.path.join(checkpoint_root, "g%d" % i)
        sess = tf.Session()
        sess_init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(sess_init_op)
        saver.restore(sess, tf.train.latest_checkpoint(check_point + '/'))
        print("restore model %d...Done!" % i)
        sessions.append(sess)

    while not net.close:
        if len(np.shape(net.evaluate_image)) < 4:
            net.evaluate_image = np.reshape(net.evaluate_image, [1, net.params['height'], net.params['width'], net.params['depth']])
        if len(np.shape(net.roi_box)) < 2:
            net.roi_box = np.reshape(net.roi_box, [1, 4])

        preds = []
        probs = []  # correct and incorrect probability
        for i in range(K):
            pred, prob = sessions[i].run([net.pred, probability], feed_dict={input: net.evaluate_image, roi_box: net.roi_box})
            pred = pred[0]
            prob = prob[0]
            preds.append(pred)
            probs.append(prob)
        yield preds, probs

    '''close'''
    for sess in sessions:
        sess.close()

'''
Evaluate tfrecord
'''
def BatchTest(checkpoint_root, testing_tfrecord_filename, K, net, Alpha, is_training=False):
    input = tf.placeholder(tf.float32, [None, net.params['height'], net.params['width'], net.params['depth']])
    roi_box = tf.placeholder(tf.float32, [None, 4])

    logits = net._inference(input, roi_box, is_training)
    probability = tf.nn.softmax(logits)

    '''restore sessions'''
    sessions = []
    saver = tf.train.Saver(max_to_keep=2)
    for i in range(K):
        check_point = os.path.join(checkpoint_root, "g%d" % i)
        sess = tf.Session()
        sess_init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(sess_init_op)
        saver.restore(sess, tf.train.latest_checkpoint(check_point + '/'))
        print("restore model %d...Done!" % i)
        sessions.append(sess)

    record_iterator = tf.python_io.tf_record_iterator(path=testing_tfrecord_filename)

    # result of ensembly (tp[0]), learner1 (tp[1]), learner2 (tp[2]), ...
    tp = np.zeros(K+1, dtype=np.int32)
    tn = np.zeros(K+1, dtype=np.int32)
    fp = np.zeros(K+1, dtype=np.int32)
    fn = np.zeros(K+1, dtype=np.int32)

    tp_prob = np.zeros(K + 1, dtype=np.int32)
    tn_prob = np.zeros(K + 1, dtype=np.int32)
    fp_prob = np.zeros(K + 1, dtype=np.int32)
    fn_prob = np.zeros(K + 1, dtype=np.int32)

    count=0
    for string_record in record_iterator:
        count+=1
        progress = "evaluate count %d" % (count)
        sys.stdout.write('\r' + progress)

        example = tf.train.Example()
        example.ParseFromString(string_record)
        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])
        # id = int(example.features.feature['id'].int64_list.value[0])
        img_string = example.features.feature['training_input'].bytes_list.value[0]
        img_1d = np.fromstring(img_string, dtype=np.uint8)
        img = img_1d.reshape((height, width, -1))
        input_img = img.astype(np.float32)
        input_img = np.reshape(input_img, [1, height, width, 3])
        training_target = example.features.feature['training_target'].float_list.value
        training_roi = example.features.feature['training_roi'].float_list.value
        input_roi = np.reshape(training_roi, [1, 4])

        gt = np.argmax(training_target)
        preds = []
        probs = []          # correct and incorrect probability
        for i in range(len(sessions)):
            pred, prob = sessions[i].run([net.pred, probability], feed_dict={input: input_img, roi_box: input_roi})
            pred=pred[0]
            prob=prob[0]
            if pred == gt and pred == 1:
                tp[i+1] += 1
            if pred == gt and pred == 0:
                tn[i+1] += 1
            if pred != gt and pred == 1:
                fp[i+1] += 1
            if pred != gt and pred == 0:
                fn[i+1] += 1

            # f[i + 1].write("%f\t%d\n" % (prob[1], gt))

            if pred==1:
                preds.append(1)
            else:
                preds.append(-1)
            probs.append(prob)


        '''ensemble result'''
        correct_probs = []
        for i in range(len(probs)):
            correct_probs.append(probs[i][1])
        correct_probability = np.sum(np.multiply(correct_probs, Alpha)) / np.sum(Alpha)
        if correct_probability>0.6:
            final_class_prob = 1
        else:
            final_class_prob = 0
        if final_class_prob==gt and final_class_prob==1:
            tp_prob[0]+=1
        if final_class_prob==gt and final_class_prob==0:
            tn_prob[0]+=1
        if final_class_prob!=gt and final_class_prob==1:
            fp_prob[0]+=1
        if final_class_prob!=gt and final_class_prob==0:
            fn_prob[0]+=1

        sign = np.sum(np.multiply(preds, Alpha))
        if sign>0:
            final_class = 1
        else:
            final_class = 0
        if final_class==gt and final_class==1:
            tp[0]+=1
        if final_class==gt and final_class==0:
            tn[0]+=1
        if final_class!=gt and final_class==1:
            fp[0]+=1
        if final_class!=gt and final_class==0:
            fn[0]+=1

        # f[0].write("%f\t%d\n" % (correct_probability, gt))

    print("\n")
    for i in range(K+1):
        if i==0:
            print("evaluation result for ensemble: tp,tn,fn,fp = %d, %d, %d, %d"%(tp[i], tn[i], fn[i], fp[i]))
            print("evaluation result for ensemble (prob): tp,tn,fn,fp = %d, %d, %d, %d"%(tp_prob[i], tn_prob[i], fn_prob[i], fp_prob[i]))
        else:
            print("evaluation result for learner%d: tp,tn,fn,fp = %d, %d, %d, %d" % (i, tp[i], tn[i], fn[i], fp[i]))

    '''close'''
    for sess in sessions:
        sess.close()


def ValidatePathNet(alignments, gt_pose, fragments_dir, net, evaluator, K, Alpha, bg_color, save_all_leaner=False):
    r_err_threshold = 4
    t_err_threshold = 50

    # result of ensembly (tp[0]), learner1 (tp[1]), learner2 (tp[2]), ...
    tp = np.zeros(K + 1, dtype=np.int32)
    tn = np.zeros(K + 1, dtype=np.int32)
    fp = np.zeros(K + 1, dtype=np.int32)
    fn = np.zeros(K + 1, dtype=np.int32)

    # create filtered results
    if save_all_leaner:
        f = []
        for i in range(K+1):
            f.append(open(os.path.join(fragments_dir, "scored_alignments_%d.txt"%i), 'w'))
            f[i].write("# v1, v2, correct_probability, gt, final_class\n# transformation matrix\n")
    else:
        f1 = open(os.path.join(fragments_dir, "filtered_alignments.txt"), 'w')

    for alignment in alignments.data:
        v1 = alignment.frame1
        v2 = alignment.frame2
        rank = alignment.rank
        trans = alignment.transform
        raw_stitch_line = alignment.stitchLine

        # gt judgement
        gt = 0
        pose1 = gt_pose.data[v1]
        pose2 = gt_pose.data[v2]
        gt_trans = np.matmul(np.linalg.inv(pose1), pose2)
        err_trans = np.matmul(trans, np.linalg.inv(gt_trans))
        if np.abs(err_trans[0, 0] - 1) < 1e-3:
            err_trans[0, 0] = 1
        if np.abs(err_trans[0, 0] + 1) < 1e-3:
            err_trans[0, 0] = -1
        theta = np.arccos(err_trans[0, 0]) * 180 / 3.1415926
        translation_err = np.sqrt(err_trans[0, 2] ** 2 + err_trans[1, 2] ** 2)
        if theta < r_err_threshold and translation_err < t_err_threshold:
            gt = 1

        # neural network judgement
        image1 = cv2.imread(os.path.join(fragments_dir, "fragment_{0:04}.png".format(v1 + 1)))
        image2 = cv2.imread(os.path.join(fragments_dir, "fragment_{0:04}.png".format(v2 + 1)))
        item = PairwiseAlignment2Image.FusionImage(image1, image2, trans, bg_color)
        if len(item) <= 0:
            continue
        path_img, overlap_ratio, transform_offset = item[0], item[1], item[2]
        resized_path_img = cv2.resize(path_img, (Parameters.NNHyperparameters["height"], Parameters.NNHyperparameters["width"]))

        net.evaluate_image = resized_path_img
        [new_min_row_ratio, new_min_col_ratio, new_max_row_ratio, new_max_col_ratio] = Utils.ConvertRawStitchLine2BBoxRatio(raw_stitch_line, path_img, trans, transform_offset, max_expand_threshold=32)
        net.roi_box = [new_min_row_ratio, new_min_col_ratio, new_max_row_ratio, new_max_col_ratio]
        preds, probs = next(evaluator)

        for i in range(K):
            if preds[i] == gt and preds[i] == 1:
                tp[i + 1] += 1
            if preds[i] == gt and preds[i] == 0:
                tn[i + 1] += 1
            if preds[i] != gt and preds[i] == 1:
                fp[i + 1] += 1
            if preds[i] != gt and preds[i] == 0:
                fn[i + 1] += 1

            if save_all_leaner:
                f[i + 1].write("%d\t%d\t%f\t%d\t%d\n" % (v1, v2, probs[i][1], gt, preds[i]))
                f[i + 1].write("%f %f %f\n%f %f %f\n0 0 1\n" % (trans[0, 0], trans[0, 1], trans[0, 2], trans[1, 0], trans[1, 1], trans[1, 2]))

            if preds[i]==1:
                preds[i]=1
            else:
                preds[i]=-1


        '''ensemble result'''
        correct_probs = []
        for i in range(len(probs)):
            correct_probs.append(probs[i][1])
        correct_probability = np.sum(np.multiply(correct_probs, Alpha))/np.sum(Alpha)
        sign = np.sum(np.multiply(preds, Alpha))
        if sign > 0:
            final_class = 1
        else:
            final_class = 0
        # if correct_probability>0.6:
        #     final_class = 1
        # else:
        #     final_class = 0

        if save_all_leaner:
            if final_class == gt and final_class == 1:
                tp[0] += 1
                # f1.write("%d\t%d\t%f\t0\n" % (v1, v2, correct_probability))
                # f1.write("%f %f %f\n%f %f %f\n0 0 1\n" % (trans[0, 0], trans[0, 1], trans[0, 2], trans[1, 0], trans[1, 1], trans[1, 2]))
            if final_class == gt and final_class == 0:
                tn[0] += 1
            if final_class != gt and final_class == 1:
                fp[0] += 1
                # f1.write("%d\t%d\t%f\t1\n" % (v1, v2, correct_probability))
                # f1.write("%f %f %f\n%f %f %f\n0 0 1\n" % (trans[0, 0], trans[0, 1], trans[0, 2], trans[1, 0], trans[1, 1], trans[1, 2]))
            if final_class != gt and final_class == 0:
                fn[0] += 1
            f[0].write("%d\t%d\t%f\t%d\t%d\n" % (v1, v2, correct_probability, gt, final_class))
            f[0].write("%f %f %f\n%f %f %f\n0 0 1\n" % (trans[0, 0], trans[0, 1], trans[0, 2], trans[1, 0], trans[1, 1], trans[1, 2]))
        else:
            if final_class == gt and final_class == 1:
                tp[0] += 1
                f1.write("%d\t%d\t%f\t0\n" % (v1, v2, correct_probability))
                f1.write("%f %f %f\n%f %f %f\n0 0 1\n" % (trans[0, 0], trans[0, 1], trans[0, 2], trans[1, 0], trans[1, 1], trans[1, 2]))
            if final_class == gt and final_class == 0:
                tn[0] += 1
            if final_class != gt and final_class == 1:
                fp[0] += 1
                f1.write("%d\t%d\t%f\t1\n" % (v1, v2, correct_probability))
                f1.write("%f %f %f\n%f %f %f\n0 0 1\n" % (trans[0, 0], trans[0, 1], trans[0, 2], trans[1, 0], trans[1, 1], trans[1, 2]))
            if final_class != gt and final_class == 0:
                fn[0] += 1

    if save_all_leaner:
        for i in range(len(f)):
            f[i].close()
    else:
        f1.close()

    for i in range(K + 1):
        if i == 0:
            print("evaluation result for ensemble: tp,tn,fn,fp = %d, %d, %d, %d" % (tp[i], tn[i], fn[i], fp[i]))
        else:
            print("evaluation result for learner%d: tp,tn,fn,fp = %d, %d, %d, %d" % (i, tp[i], tn[i], fn[i], fp[i]))
    return tp, tn, fn, fp


def main(_):
    mode = Args['mode']
    K = Parameters.NNHyperparameters["learner_num"]
    params = Parameters.NNHyperparameters
    checkpoint_root = Parameters.WorkSpacePath['checkpoint_dir']

    if mode == "training":
        tfrecord_filename_training = "/home/nugh75/Git/JigsawNet/train_tfrecord_fixed.tfrecord"
        tfrecord_filename_testing = "/home/nugh75/Git/JigsawNet/test_tfrecord_fixed.tfrecord"

        
        total_record = sum(1 for _ in tf.python_io.tf_record_iterator(tfrecord_filename_training))
        D = np.ones(total_record, dtype=np.float32)
        Alpha = np.zeros(K, dtype=np.float32)

        for i in range(K):
            subdir = f"g{i}/tensorboard"
            tensorboard_dir = os.path.join(checkpoint_root, subdir)
            checkpoint_dir = os.path.dirname(tensorboard_dir)
            os.makedirs(tensorboard_dir, exist_ok=True)

            np.savetxt(os.path.join(checkpoint_dir, "data_weight.txt"), D, delimiter=' ')

            net = JIgsawAbitraryNetROI.JigsawNetWithROI(params)

            # ⬇️ Training queue
            filename_queue_train = tf.train.string_input_producer([tfrecord_filename_training], capacity=128)
            inputs, targets, roi_boxes, data_ids = TFRecordIOWithROI.readTFRecord(filename_queue_train)
            weights = tf.constant(D, dtype=tf.float32)

            # ⬇️ Testing queue
            filename_queue_test = tf.train.string_input_producer([tfrecord_filename_testing], capacity=128)
            inputs_test, targets_test, roi_boxes_test, data_ids_test = TFRecordIOWithROI.readTFRecord(filename_queue_test)
            weights_test = tf.constant(1.0, dtype=tf.float32)  # dummy weight for now

            # ⬇️ Train with both training and test data
            BoostTraining(
                net=net,
                input=inputs, roi_box=roi_boxes, target=targets, weights=weights, data_ids=data_ids,
                input_test=inputs_test, roi_box_test=roi_boxes_test, target_test=targets_test, weights_test=weights_test, data_ids_test=data_ids_test,
                tensorboard_dir=tensorboard_dir,
                checkpoint_dir=checkpoint_dir,
                is_training=True
            )

            tf.reset_default_graph()

            # Evaluate model and update weights
            filename_queue_eval = tf.train.string_input_producer([tfrecord_filename_training], shuffle=False, num_epochs=1)
            inputs_eval, targets_eval, roi_boxes_eval, data_ids_eval = TFRecordIOWithROI.readTFRecord(filename_queue_eval)
            weights_eval = tf.constant(D, dtype=tf.float32)
            alpha, new_weights = Evaluation(
                net=net,
                input=inputs_eval,
                roi_box=roi_boxes_eval,
                target=targets_eval,
                weights=weights_eval,
                data_ids=data_ids_eval,
                checkpoint_dir=checkpoint_dir,
                is_training=False
            )
            Alpha[i] = alpha
            D = new_weights

            tf.reset_default_graph()

            with open(os.path.join(checkpoint_dir, "alpha.txt"), 'w') as f:
                f.write("%f" % alpha)

        with open(os.path.join(checkpoint_root, "alpha.txt"), 'w') as f:
            f.write(" ".join(f"{a:.6f}" for a in Alpha))
    elif mode == "batch_testing":           # use tfrecord as input to evaluate
        '''Batch Testing'''
        testing_directory_root = Parameters.WorkSpacePath['testing_dataset_root']
        testing_tfrecord_filename = os.path.join(Parameters.WorkSpacePath['testing_dataset_root'], 'input_tfrecord_roi')
        if not os.path.exists(testing_tfrecord_filename):
            TFRecordIOWithROI.createTFRecord(testing_tfrecord_filename, dataset_root=testing_directory_root)

        net = JIgsawAbitraryNetROI.JigsawNetWithROI(params)
        with open(os.path.join(checkpoint_root, "alpha.txt")) as f:
            for line in f:
                line = line.rstrip()
                if line[0] != '#':
                    line = line.split()
                    Alpha = [float(x) for x in line]
        BatchTest(checkpoint_root=checkpoint_root, testing_tfrecord_filename=testing_tfrecord_filename, K=K, net=net, Alpha=Alpha, is_training=False)
    elif mode == "single_testing":          # use alignment transformation as input to evaluate.
        '''Single Testing'''
        testing_data_root1 = Parameters.WorkSpacePath["example_testing_root"]
        fragments_dirs = glob.glob(os.path.join(testing_data_root1, "*_ex"))

        with open(os.path.join(checkpoint_root, "alpha.txt")) as f:
            for line in f:
                line = line.rstrip()
                if line[0] != '#':
                    line = line.split()
                    Alpha = [float(x) for x in line]

        net = JIgsawAbitraryNetROI.JigsawNetWithROI(params=Parameters.NNHyperparameters)
        evaluator = SingleTest(checkpoint_root=checkpoint_root, K=5, net=net, is_training=False)

        for i in range(len(fragments_dirs)):
            print("dataset %d/%d:" % (i, len(fragments_dirs)))
            if not os.path.exists(os.path.join(fragments_dirs[i], "alignments.txt")):
                continue
            bg_color_file = os.path.join(fragments_dirs[i], "bg_color.txt")
            with open(bg_color_file) as f:
                for line in f:
                    line = line.split()
                    if line:
                        bg_color = [int(i) for i in line]
                        bg_color = bg_color[::-1]
            gt_pose = os.path.join(fragments_dirs[i], "groundTruth.txt")
            relative_alignment = os.path.join(fragments_dirs[i], "alignments.txt")
            gt_pose = Utils.GtPose(gt_pose)
            alignments = Utils.Alignment2d(relative_alignment)
            ValidatePathNet(alignments, gt_pose, fragments_dirs[i], net, evaluator, K, Alpha, bg_color, save_all_leaner=False)
            print("----------------")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--mode',
        help="Choose a net running mode: training, batch_testing or single_testing",
        default="training"  # ✅ Default for IDE use
    )
    Args = vars(parser.parse_args())

    tf.compat.v1.disable_eager_execution()  # ✅ Ensure TF1 compatibility
    tf.compat.v1.app.run(main=main)