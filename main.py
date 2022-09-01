import argparse
import os
import datetime
import time
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from decoder.utils import convert_cycle

from model import DGMG_VAE

def main(opts):
    t1 = time.time()

    # Setup dataset and data loader
    if opts['dataset'] == 'cycles':
        from data.cycles import CycleDataset, CycleModelEvaluation, CyclePrinting

        dataset = CycleDataset(fname=opts['path_to_dataset'])
        evaluator = CycleModelEvaluation(v_min=opts['min_size'],
                                         v_max=opts['max_size'],
                                         dir=opts['log_dir'])
        printer = CyclePrinting(num_epochs=opts['nepochs'],
                                num_batches=opts['ds_size'] // opts['batch_size'])
    else:
        raise ValueError('Unsupported dataset: {}'.format(opts['dataset']))

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                             collate_fn=dataset.collate_single)

    # Initialize_model
    model = DGMG_VAE(opts)
    
    # Initialize optimizer
    if opts['optimizer'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=opts['lr'])
    else:
        raise ValueError('Unsupported argument for the optimizer')

    t2 = time.time()

    # Training
    
    for epoch in range(opts['nepochs']):
        model.train()
        batch_count = 0
        batch_loss = 0
        optimizer.zero_grad()

        for i, data in enumerate(tqdm(data_loader)):
            x, edge_index = convert_cycle(data)
            loss = model(x, edge_index, actions=data) # train on data
            loss.backward() # backpropagate

            batch_loss += loss.item()
            #batch_prob += prob_averaged.item()
            batch_count += 1
 
            if batch_count % opts['batch_size'] == 0:
                print('\n')
                printer.update(epoch + 1, {'averaged_loss': batch_loss/opts['batch_size']})

                if opts['clip_grad']:
                    clip_grad_norm_(model.parameters(), opts['clip_bound'])

                optimizer.step()

                batch_loss = 0
                #
                optimizer.zero_grad()
        model.eval()
        evaluator.rollout_and_examine(model, opts['num_generated_samples'])

    t3 = time.time()

    model.eval()
    evaluator.rollout_and_examine(model, opts['num_generated_samples'])
    evaluator.write_summary()

    t4 = time.time()

    print('It took {} to setup.'.format(datetime.timedelta(seconds=t2-t1)))
    print('It took {} to finish training.'.format(datetime.timedelta(seconds=t3-t2)))
    print('It took {} to finish evaluation.'.format(datetime.timedelta(seconds=t4-t3)))
    print('--------------------------------------------------------------------------')
    #print('On average, an epoch takes {}.'.format(datetime.timedelta(
    #    seconds=(t3-t2) / opts['nepochs'])))

    del model.g
    torch.save(model, './model.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')

    # configure
    parser.add_argument('--seed', type=int, default=9284, help='random seed')

    #encoder
    parser.add_argument('--enc_in_channels', type=int, default=1433, help='in_channels')
    parser.add_argument('--enc_hidden_channels', type=int, default=32, help='hidden_channels')
    parser.add_argument('--enc_out_channels', type=int, default=16, help='out_channels')
    

    # dataset
    parser.add_argument('--dataset', choices=['cycles'], default='cycles',
                        help='dataset to use')
    parser.add_argument('--path-to-dataset', type=str, default='./data/cycles.p',
                        help='load the dataset if it exists, '
                             'generate it and save to the path otherwise')

    # log
    parser.add_argument('--log-dir', default='./results',
                        help='folder to save info like experiment configuration '
                             'or model evaluation results')

    # optimization
    parser.add_argument('--batch-size', type=int, default=20,
                        help='batch size to use for training')
    parser.add_argument('--clip-grad', action='store_true', default=True,
                        help='gradient clipping is required to prevent gradient explosion')
    parser.add_argument('--clip-bound', type=float, default=0.25,
                        help='constraint of gradient norm for gradient clipping')
    parser.add_argument('--reg', type=float, default=1, help='regularization for KL loss')
    parser.add_argument('--nepochs', type=int, default=5, help='number of epochs for training')
    args = parser.parse_args()
    
    from decoder.utils import setup
    opts = setup(args)

    main(opts)