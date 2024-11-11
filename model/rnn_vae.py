#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 0.1 Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import torch
from torch import nn
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import copy
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

from vame.util.auxiliary import read_config
from vame.model.dataloader import SEQUENCE_DATASET
from vame.model.rnn_model import RNN_VAE, RNN_VAE_LEGACY

# make sure torch uses cuda for GPU computing
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")
    print('GPU active:',torch.cuda.is_available())
    print('GPU used:',torch.cuda.get_device_name(0))
else:
    torch.device("cpu")

def gaussian_kernel(z, z_prior):
    dim1_1, dim1_2 = z.shape[0], z_prior.shape[0]
    depth = z.shape[1]
    z = z.view(dim1_1, 1, depth)
    z_prior = z_prior.view(1, dim1_2, depth)
    z_core = z.expand(dim1_1, dim1_2, depth)
    z_prior_core = z_prior.expand(dim1_1, dim1_2, depth)
    numerator = (z_core - z_prior_core).pow(2).mean(2)/depth
    return torch.exp(-numerator)

def max_mean_discrapency_loss(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    #z = mu + eps*std
    #z = copy.deepcopy(mu)
    z = mu
    z_prior = torch.randn_like(z)
    return gaussian_kernel(z, z).mean() + gaussian_kernel(z_prior, z_prior).mean() - 2*gaussian_kernel(z, z_prior).mean()

def reconstruction_loss(x, x_tilde, reduction):
    mse_loss = nn.MSELoss(reduction=reduction)
    rec_loss = mse_loss(x_tilde,x)
    return rec_loss

def future_reconstruction_loss(x, x_tilde, reduction):
    mse_loss = nn.MSELoss(reduction=reduction)
    rec_loss = mse_loss(x_tilde,x)
    return rec_loss


def cluster_loss(H, kloss, lmbda, batch_size):
    gram_matrix = (H.T @ H) / batch_size
    _ ,sv_2, _ = torch.svd(gram_matrix)
    sv = torch.sqrt(sv_2[:kloss])
    loss = torch.sum(sv)
    return lmbda*loss


def kullback_leibler_loss(mu, logvar):
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


def kl_annealing(epoch, kl_start, annealtime, function):
    """
        Annealing of Kullback-Leibler loss to let the model learn first
        the reconstruction of the data before the KL loss term gets introduced.
    """
    if epoch > kl_start:
        if function == 'linear':
            new_weight = min(1, (epoch-kl_start)/(annealtime))

        elif function == 'sigmoid':
            new_weight = float(1/(1+np.exp(-0.9*(epoch-annealtime))))
        else:
            raise NotImplementedError('currently only "linear" and "sigmoid" are implemented')

        return new_weight

    else:
        new_weight = 0
        return new_weight


def gaussian(ins, is_training, seq_len, std_n=0.8):
    if is_training:
        emp_std = ins.std(1)*std_n
        emp_std = emp_std.unsqueeze(2).repeat(1, 1, seq_len)
        emp_std = emp_std.permute(0,2,1)
        noise = Variable(ins.data.new(ins.size()).normal_(0, 1))
        return ins + (noise*emp_std)
    return ins


def train(train_loader, epoch, model, optimizer, anneal_function, BETA, kl_start,
          annealtime, seq_len, future_decoder, future_steps, scheduler, mse_red, 
          mse_pred, kloss, klmbda, bsize, noise):
    model.train() 
    train_loss = 0.0
    mse_loss = 0.0
    kullback_loss = 0.0
    mmd_losses = 0.0
    kmeans_losses = 0.0
    vel_loss = 0.0
    fut_loss = 0.0
    loss = 0.0
    seq_len_half = int(seq_len / 2)
    
    for idx, data_item in enumerate(train_loader):
        velocity = Variable(data_item['velocity'])
        frames = Variable(data_item['frames'])
        data_item = Variable(data_item['sequence'])
        data_item = data_item.permute(0,2,1)
        velocity = velocity.permute(0,2,1)
        frames = frames.permute(0,2,1)
        if use_gpu:
            data = data_item[:,:seq_len_half,:].type('torch.FloatTensor').cuda()
            vel = velocity[:, :seq_len_half, :].type('torch.FloatTensor').cuda()
            fr = frames[:, :seq_len_half, :].type('torch.FloatTensor').cuda()

            fut = data_item[:,seq_len_half:seq_len_half+future_steps,:].type('torch.FloatTensor').cuda()
        else:
            data = data_item[:,:seq_len_half,:].type('torch.FloatTensor').to()
            vel = velocity[:, :seq_len_half, :].type('torch.FloatTensor').to()
            fr = frames[:, :seq_len_half, :].type('torch.FloatTensor').to()
            fut = data_item[:,seq_len_half:seq_len_half+future_steps,:].type('torch.FloatTensor').to()
        if noise == True:
            data_gaussian = gaussian(data,True,seq_len_half)
        else:
            data_gaussian = data

        if future_decoder:
            data_tilde, future, vel_tilde, latent, mu, logvar = model(data_gaussian)
           
            rec_loss = reconstruction_loss(data, data_tilde, mse_red)
            fut_rec_loss = future_reconstruction_loss(fut, future, mse_pred)
            vel_rec_loss = reconstruction_loss(vel, vel_tilde, mse_red)
            kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
            mmd_loss = max_mean_discrapency_loss(mu, logvar)
            kl_loss = kullback_leibler_loss(mu, logvar)
            kl_weight = kl_annealing(epoch, kl_start, annealtime, anneal_function)
            loss = rec_loss + fut_rec_loss + kl_weight*kmeans_loss + BETA*kl_weight*kl_loss#+ kl_weight*mmd_loss
            fut_loss += fut_rec_loss.item()

        else:
            data_tilde, vel_tilde,latent, mu, logvar = model(data_gaussian)
            mmd_loss = max_mean_discrapency_loss(mu, logvar)
            rec_loss = reconstruction_loss(data, data_tilde, mse_red)
            vel_rec_loss = reconstruction_loss(vel, vel_tilde, mse_red)
            kl_loss = kullback_leibler_loss(mu, logvar)
            kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
            kl_weight = kl_annealing(epoch, kl_start, annealtime, anneal_function)
            loss = rec_loss + kl_weight*kmeans_loss + BETA*kl_weight*kl_loss #kl_weight*
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 5)
        
        train_loss += loss.item()
        mse_loss += rec_loss.item()
        kullback_loss += kl_loss.item()
        kmeans_losses += kmeans_loss.item()
        mmd_losses += mmd_loss.item()
        vel_loss += vel_rec_loss.item()
        # if idx % 1000 == 0:
        #     print('Epoch: %d.  loss: %.4f' %(epoch, loss.item()))
   
    scheduler.step(loss) #be sure scheduler is called before optimizer in >1.1 pytorch

    if future_decoder:
        print('Train loss: {:.3f}, MSE-Loss: {:.3f}, MSE-Future-Loss {:.3f}, Velocity-Loss: {:.3f}, MMD-Loss: {:.3f}, Kmeans-Loss: {:.3f}, weight: {:.2f}'.format(train_loss / idx,
              mse_loss /idx, fut_loss/idx, 0.000 , BETA*kl_weight*kullback_loss/idx, kl_weight*kmeans_losses/idx, kl_weight))
    else:
        print('Train loss: {:.3f}, MSE-Loss: {:.3f}, MMD-Loss: {:.3f}, Velocity-Loss: {:.3f}, Kmeans-Loss: {:.3f}, weight: {:.2f}'.format(train_loss / idx,
              mse_loss /idx, BETA*kl_weight*kullback_loss/idx, 0.000 ,kl_weight*kmeans_losses/idx, kl_weight))

    return kl_weight, train_loss/idx, kl_weight*kmeans_losses/idx, kullback_loss/idx, mse_loss/idx, fut_loss/idx, mmd_losses/idx


def test(test_loader, epoch, model, optimizer, BETA, kl_weight, seq_len, mse_red, kloss, klmbda, future_decoder, future_steps,bsize):
    model.eval() # toggle model to inference mode
    test_loss = 0.0
    mse_loss = 0.0
    kullback_loss = 0.0
    kmeans_losses = 0.0
    mmd_losses = 0.0
    vel_loss = 0.0
    loss = 0.0
    seq_len_half = int(seq_len / 2)

    with torch.no_grad():
        for idx, data_item in enumerate(test_loader):

            velocity = Variable(data_item['velocity'])
            data_item = Variable(data_item['sequence'])
            data_item = data_item.permute(0,2,1)
            velocity = velocity.permute(0,2,1)
            if use_gpu:
                data = data_item[:,:seq_len_half,:].type('torch.FloatTensor').cuda()
                vel = velocity[:, :seq_len_half, :].type('torch.FloatTensor').cuda()
                fut = data_item[:,seq_len_half:seq_len_half+future_steps,:].type('torch.FloatTensor').cuda()
            else:
                data = data_item[:,:seq_len_half,:].type('torch.FloatTensor').to()
                vel = velocity[:, :seq_len_half, :].type('torch.FloatTensor').to()
                fut = data_item[:,seq_len_half:seq_len_half+future_steps,:].type('torch.FloatTensor').to()

            if future_decoder:
                data_tilde, fut_tilde, vel_tilde, latent, mu, logvar = model(data)
                rec_loss = reconstruction_loss(data, data_tilde, mse_red)
                fut_rec_loss = reconstruction_loss(fut, fut_tilde, mse_red)
                kl_loss = kullback_leibler_loss(mu, logvar)
                kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
                mmd_loss = max_mean_discrapency_loss(mu, logvar)
                loss = rec_loss + kl_weight*kmeans_loss + BETA*kl_weight*kl_loss 

            else:
                data_tilde, vel_tilde, latent, mu, logvar = model(data)
                rec_loss = reconstruction_loss(data, data_tilde, mse_red)
                vel_rec_loss = reconstruction_loss(vel, vel_tilde, mse_red)
                kl_loss = kullback_leibler_loss(mu, logvar)
                kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
                mmd_loss = max_mean_discrapency_loss(mu, logvar)
                loss = rec_loss + kl_weight*kmeans_loss + BETA*kl_weight*kl_loss #+

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 5)

            test_loss += loss.item()
            mse_loss += rec_loss.item()
            kullback_loss += kl_loss.item()
            kmeans_losses += kmeans_loss
            mmd_losses += mmd_loss.item()
            #vel_loss += vel_rec_loss.item()

    print('Test loss: {:.3f}, MSE-Loss: {:.3f}, KL-Loss: {:.3f}, Velocity-Loss: {:.3f}, Kmeans-Loss: {:.3f}'.format(test_loss / idx,
          mse_loss /idx, BETA*kl_weight*kullback_loss/idx, 0.000, kl_weight*kmeans_losses/idx))

    return mse_loss /idx, test_loss/idx, kl_weight*kmeans_losses


def train_model(config):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    legacy = cfg['legacy']
    model_name = cfg['model_name']
    pretrained_weights = cfg['pretrained_weights']
    pretrained_model = cfg['pretrained_model']
    fixed = cfg['egocentric_data']
    
    print("Train Variational Autoencoder - model name: %s \n" %model_name)
    if not os.path.exists(os.path.join(cfg['project_path'],'model','best_model',"")):
        os.mkdir(os.path.join(cfg['project_path'],'model','best_model',""))
        os.mkdir(os.path.join(cfg['project_path'],'model','best_model','snapshots',""))
        os.mkdir(os.path.join(cfg['project_path'],'model','model_losses',""))

    # make sure torch uses cuda for GPU computing
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")
        print('GPU active:',torch.cuda.is_available())
        print('GPU used: ',torch.cuda.get_device_name(0))
    else:
        torch.device("cpu")
        print("warning, a GPU was not found... proceeding with CPU (slow!) \n")
        #raise NotImplementedError('GPU Computing is required!')
        
    """ HYPERPARAMTERS """
    # General
    CUDA = use_gpu
    SEED = 19
    TRAIN_BATCH_SIZE = cfg['batch_size']
    TEST_BATCH_SIZE = int(cfg['batch_size']/4)
    EPOCHS = cfg['max_epochs']
    ZDIMS = cfg['zdims']
    BETA  = cfg['beta']
    SNAPSHOT = cfg['model_snapshot']
    LEARNING_RATE = cfg['learning_rate']
    NUM_FEATURES = cfg['num_features']
    if fixed == False:
        NUM_FEATURES = NUM_FEATURES - 2
    TEMPORAL_WINDOW = cfg['time_window']*2
    FUTURE_DECODER = cfg['prediction_decoder']
    FUTURE_STEPS = cfg['prediction_steps']

    # RNN
    hidden_size_layer_1 = cfg['hidden_size_layer_1']
    hidden_size_layer_2 = cfg['hidden_size_layer_2']
    hidden_size_rec = cfg['hidden_size_rec']
    hidden_size_pred = cfg['hidden_size_pred']
    dropout_encoder = cfg['dropout_encoder']
    dropout_rec = cfg['dropout_rec']
    dropout_pred = cfg['dropout_pred']
    noise = cfg['noise']
    scheduler_step_size = cfg['scheduler_step_size']
    softplus = cfg['softplus']

    # Loss
    MSE_REC_REDUCTION = cfg['mse_reconstruction_reduction']
    MSE_PRED_REDUCTION = cfg['mse_prediction_reduction']
    KMEANS_LOSS = cfg['kmeans_loss']
    KMEANS_LAMBDA = cfg['kmeans_lambda']
    KL_START = cfg['kl_start']
    ANNEALTIME = cfg['annealtime']
    anneal_function = cfg['anneal_function']
    optimizer_scheduler = cfg['scheduler']

    BEST_LOSS = 999999
    convergence = 0
    print('Latent Dimensions: %d, Time window: %d, Batch Size: %d, Beta: %d, lr: %.4f\n' %(ZDIMS, cfg['time_window'], TRAIN_BATCH_SIZE, BETA, LEARNING_RATE))
    
    # simple logging of diverse losses
    train_losses = []
    test_losses = []
    kmeans_losses = []
    kl_losses = []
    weight_values = []
    mse_losses = []
    mmd_losses = []
    fut_losses = []

    torch.manual_seed(SEED)
    
    if legacy == False:
        RNN = RNN_VAE
    else:
        RNN = RNN_VAE_LEGACY
    if CUDA:
        torch.cuda.manual_seed(SEED)
        model = RNN(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1,
                        hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder,
                        dropout_rec, dropout_pred, softplus).cuda()
    else: 
        torch.cuda.manual_seed(SEED)
        model = RNN(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1,
                        hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder,
                        dropout_rec, dropout_pred, softplus).to()

    if pretrained_weights:
        if os.path.exists(os.path.join(cfg['project_path'],'model','best_model',pretrained_model+'_'+cfg['Project']+'.pkl')): 
            print("Loading pretrained weights from model: %s\n" %pretrained_model)
            model.load_state_dict(torch.load(os.path.join(cfg['project_path'],'model','best_model',pretrained_model+'_'+cfg['Project']+'.pkl')))
            KL_START = 0
            ANNEALTIME = 1
            
    """ DATASET """
    trainset = SEQUENCE_DATASET(os.path.join(cfg['project_path']), data=['train_velocity_seq.npy','train_seq.npy'], train=True, temporal_window=TEMPORAL_WINDOW)
    testset = SEQUENCE_DATASET(os.path.join(cfg['project_path']), data=['test_velocity_seq.npy','test_seq.npy'], train=False, temporal_window=TEMPORAL_WINDOW)
    print(len(trainset))
  

    train_loader = Data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = Data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)

    if optimizer_scheduler:
        print('Scheduler step size: %d, Scheduler gamma: %.2f\n' %(scheduler_step_size, cfg['scheduler_gamma']))
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=cfg['scheduler_gamma'], patience=cfg['scheduler_step_size'], threshold=1e-3, threshold_mode='rel', verbose=True)
    else:
        scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=1, last_epoch=-1)
    
    print("Start training... ")
    for epoch in range(1,EPOCHS):
        print("Epoch: %d" %epoch)
        trainset.idx = 0
        testset.idx = 0
        weight, train_loss, km_loss, kl_loss, mse_loss, fut_loss, mmd_loss = train(train_loader, epoch, model, optimizer,
                                                                         anneal_function, BETA, KL_START,
                                                                         ANNEALTIME, TEMPORAL_WINDOW, FUTURE_DECODER,
                                                                         FUTURE_STEPS, scheduler, MSE_REC_REDUCTION,
                                                                         MSE_PRED_REDUCTION, KMEANS_LOSS, KMEANS_LAMBDA,
                                                                         TRAIN_BATCH_SIZE, noise)

        current_loss, test_loss, test_list = test(test_loader, epoch, model, optimizer,
                                                  BETA, weight, TEMPORAL_WINDOW, MSE_REC_REDUCTION,
                                                  KMEANS_LOSS, KMEANS_LAMBDA, FUTURE_DECODER, FUTURE_STEPS,TEST_BATCH_SIZE)

        # logging losses
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        kmeans_losses.append(km_loss)
        mmd_losses.append(mmd_loss)
        kl_losses.append(kl_loss)
        weight_values.append(weight)
        mse_losses.append(mse_loss)
        fut_losses.append(fut_loss)

        # save best model
        if weight > 0.99 and test_loss <= BEST_LOSS:
            BEST_LOSS = test_loss
            print("Saving model!")

            if use_gpu:
                torch.save(model.state_dict(), os.path.join(cfg['project_path'],"model", "best_model",model_name+'_'+cfg['Project']+'.pkl'))

            else:
                torch.save(model.state_dict(), os.path.join(cfg['project_path'],"model", "best_model",model_name+'_'+cfg['Project']+'.pkl'))

            convergence = 0
        else:
            convergence += 1

        if epoch % SNAPSHOT == 0:
            print("Saving model snapshot!\n")
            torch.save(model.state_dict(), os.path.join(cfg['project_path'],'model','best_model','snapshots',model_name+'_'+cfg['Project']+'_epoch_'+str(epoch)+'.pkl'))

        if convergence > cfg['model_convergence']:
            print('Finished training...')
            print('Model converged. Please check your model with vame.evaluate_model(). \n'
                  'You can also re-run vame.trainmodel() to further improve your model. \n'
                  'Make sure to set _pretrained_weights_ in your config.yaml to "true" \n'
                  'and plug your current model name into _pretrained_model_. \n'
                  'Hint: Set "model_convergence" in your config.yaml to a higher value. \n'
                  '\n'
                  'Next: \n'
                  'Use vame.pose_segmentation() to identify behavioral motifs in your dataset!')
            #return
            break

        # save logged losses
        np.save(os.path.join(cfg['project_path'],'model','model_losses','train_losses_'+model_name), train_losses)
        np.save(os.path.join(cfg['project_path'],'model','model_losses','test_losses_'+model_name), test_losses)
        np.save(os.path.join(cfg['project_path'],'model','model_losses','kmeans_losses_'+model_name), kmeans_losses)
        np.save(os.path.join(cfg['project_path'],'model','model_losses','kl_losses_'+model_name), kl_losses)
        np.save(os.path.join(cfg['project_path'],'model','model_losses','weight_values_'+model_name), weight_values)
        np.save(os.path.join(cfg['project_path'],'model','model_losses','mse_train_losses_'+model_name), mse_losses)
        np.save(os.path.join(cfg['project_path'],'model','model_losses','mse_test_losses_'+model_name), current_loss)
        np.save(os.path.join(cfg['project_path'],'model','model_losses','fut_losses_'+model_name), fut_losses)
        np.save(os.path.join(cfg['project_path'],'model','model_losses','mmd_losses_'+model_name), mmd_losses)
        
        print("\n")

    if convergence < cfg['model_convergence']:
        print('Finished training...')
        print('Model seems to have not reached convergence. You may want to check your model \n'
              'with vame.evaluate_model(). If your satisfied you can continue. \n'
              'Use vame.pose_segmentation() to identify behavioral motifs! \n'
              'OPTIONAL: You can re-run vame.train_model() to improve performance.')
