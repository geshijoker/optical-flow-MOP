import os
import copy

import tensorflow as tf

def main(argv=None):

	cconfig = copy.deepcopy(experiment.config['train'])
    cconfig.update(experiment.config['train_chairs'])
    convert_input_strings(cconfig, dirs)
    citers = cconfig.get('num_iters', 0)
    cdata = ChairsData(data_dir=dirs['data'],
                       fast_dir=dirs.get('fast'),
                       stat_log_dir=None,
                       development=run_config['development'])
    cinput = ChairsInput(data=cdata,
             batch_size=gpu_batch_size,
             normalize=False,
             dims=(cconfig['height'], cconfig['width']))
    tr = Trainer(
          lambda shift: cinput.input_raw(swap_images=False,
                                         shift=shift * run_config['batch_size']),
          lambda: einput.input_train_2012(),
          params=cconfig,
          normalization=cinput.get_normalization(),
          train_summaries_dir=experiment.train_dir,
          eval_summaries_dir=experiment.eval_dir,
          experiment=FLAGS.ex,
          ckpt_dir=experiment.save_dir,
          debug=FLAGS.debug,
          interactive_plot=run_config.get('interactive_plot'),
          devices=devices)
    tr.run(0, citers)

    if not FLAGS.debug:
        experiment.conclude()

if __name__ == '__main__':
    tf.app.run()