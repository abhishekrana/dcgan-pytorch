import numpy as np
import torch
import utils.util as util
import time
import os
import copy
import pickle
import pudb
import glob
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
import logging

class Trainer():
    def __init__(self, config, dataloaders, model):
        self.config = config
        self.dataloaders = dataloaders
        self.model = model
        # self.scheduler = model.exp_lr_scheduler
        self.epochs = config['trainer']['epochs']

        self.writer = config['writer']
        self.device = config['device']
        self.batch_size = config['data_handler']['batch_size']
        self.dataset_sizes = config['data_handler']['dataset_sizes']
        self.nz = self.config['nz']

        self.fixed_noise = torch.randn(self.batch_size, self.nz, 1, 1, device=self.device)
        self.real_label = 1
        self.fake_label = 0


    def train(self):
        for epoch in range(self.epochs):
            for i, data in enumerate(self.dataloaders['train'], 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with real
                self.model.netD.zero_grad()
                real_cpu = data[0].to(self.device)
                batch_size = real_cpu.size(0)
                label = torch.full((batch_size,), self.real_label, device=self.device)

                output = self.model.netD(real_cpu)
                errD_real = self.model.criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with fake
                noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
                fake = self.model.netG(noise)
                label.fill_(self.fake_label)
                output = self.model.netD(fake.detach())
                errD_fake = self.model.criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                self.model.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.model.netG.zero_grad()
                label.fill_(self.real_label)  # fake labels are real for generator cost
                output = self.model.netD(fake)
                errG = self.model.criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.model.optimizerG.step()
                
                # TODO: dataloader
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, self.epochs, i, len(self.dataloaders['train']),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                if i % 100 == 0:
                    vutils.save_image(real_cpu,
                            '%s/real_samples.png' % self.config['output_images_path'],
                            normalize=True)
                    fake = self.model.netG(self.fixed_noise)
                    vutils.save_image(fake.detach(),
                            '%s/fake_samples_epoch_%03d.png' % (self.config['output_images_path'], epoch),
                            normalize=True)

            # Save checkpoints
            torch.save(self.model.netG.state_dict(), '%s/netG_epoch_%d.pth' % (self.config['checkpoint_path'], epoch))
            torch.save(self.model.netD.state_dict(), '%s/netD_epoch_%d.pth' % (self.config['checkpoint_path'], epoch))




    # def train2(self):
    #     since = time.time()

    #     best_model_wts = copy.deepcopy(self.model.state_dict())
    #     best_acc = 0.0
    #     best_loss = 100000.0

    #     for epoch in range(self.epochs):
    #         logging.debug('Epoch {}/{}'.format(epoch, self.epochs - 1))
    #         logging.debug('-' * 10)

    #         ## Each epoch has a training and validation phase
    #         for phase in ['train', 'val', 'test']:
    #             if phase == 'train':
    #                 self.scheduler.step()
    #                 self.model.train()  # Set model to training mode
    #             else:
    #                 self.model.eval()   # Set model to evaluate mode

    #             running_loss = 0.0
    #             running_corrects = 0
    #             running_images = 0

    #             ## Iterate over data.
    #             for step, (inputs, labels) in enumerate(self.dataloaders[phase]):

    #                 inputs = inputs.to(self.device)
    #                 labels = labels.to(self.device)

    #                 ## zero the parameter gradients
    #                 self.optimizer.zero_grad()

    #                 ## forward
    #                 ## track history if only in train
    #                 with torch.set_grad_enabled(phase == 'train'):
    #                     outputs = self.model(inputs)
    #                     # Returns the maximum value of each row of the input tensor in the given dimension dim. 
    #                     # The second return value is the index location of each maximum value found (argmax).
    #                     _, preds = torch.max(outputs, 1)
    #                     loss = self.criterion(outputs, labels)

    #                     ## backward + optimize only if in training phase
    #                     if phase == 'train':
    #                         loss.backward()
    #                         self.optimizer.step()

    #                 ## Statistics
    #                 running_loss += loss.item() * inputs.size(0)
    #                 running_corrects += torch.sum(preds == labels.data)

    #                 running_images = (step+1)*self.batch_size
    #                 logging.debug('{} Step:{}/{} Loss: {:.4f} Acc: {:.4f}'.format(
    #                     phase, step, int(self.dataset_sizes[phase]/self.batch_size), 
    #                     running_loss/running_images, running_corrects.double()/running_images))
    #                 # logging.debug('Step:', step, '/', int(self.dataset_sizes[phase]/self.batch_size), 
    #                 #       running_loss/running_images, running_corrects.double()/running_corrects)

    #             epoch_loss = running_loss / self.dataset_sizes[phase]
    #             epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
    #             self.writer.add_scalar(phase + '/Loss/epoch_loss', epoch_loss, epoch)
    #             self.writer.add_scalar(phase + '/Accuracy/epoch_accuracy', epoch_acc, epoch)


    #             logging.debug('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    #             ## Deep copy the model
    #             if phase == 'val':
    #                 if epoch_acc > best_acc:
    #                     best_acc = epoch_acc
    #                     best_model_prefix = 'acc'
    #                 if epoch_loss < best_loss:
    #                     best_loss = epoch_loss
    #                     best_model_prefix = 'loss'

    #                 best_model_wts = copy.deepcopy(self.model.state_dict())
    #                 best_model_name = best_model_prefix + '-epoch-'+str(epoch) + '_loss-'+str(np.round(epoch_loss, 3)) + '_acc-'+str(np.round(epoch_acc.cpu().numpy(), 3))
    #                 with open(os.path.join(self.config['checkpoint_path'], 'model-' + best_model_name + '.pkl'), 'wb') as fp:
    #                     pickle.dump(best_model_wts, fp)
    #                 logging.debug('best_model_name {}'.format(best_model_name))

    #                 torch.save(self.model.state_dict(), os.path.join(self.config['checkpoint_path'], 'model-' + best_model_name + '.pth' ))
    #                 torch.save(self.optimizer.state_dict(), os.path.join(self.config['checkpoint_path'], 'optimizer-' + best_model_name + '.pth' ))

    #                 # model.load_state_dict(torch.load(os.path.join(self.config['checkpoint_path'], 'model-' + best_model_name + '.pth')))
    #                 # model.load_state_dict(torch.load(os.path.join(self.config['checkpoint_path'], 'optimizer-' + best_model_name + '.pth')))

    #         logging.debug(' ')

    #     self.config['best_model_checkpoint'] = best_model_wts

    #     time_elapsed = time.time() - since
    #     logging.debug('Training complete in {:.0f}m {:.0f}s'.format(
    #         time_elapsed // 60, time_elapsed % 60))
    #     logging.debug('Best val Acc: {:4f}'.format(best_acc))


    # def test(self):

    #     if 'best_model_checkpoint' in self.config:
    #         logging.debug('Loading best_model_checkpoint')
    #         model_checkpoint = self.config['best_model_checkpoint']
    #     else:
    #         # with open(os.path.join(self.config.['checkpoint_path'], self.config['checkpoint_file']), 'rb') as fp:
    #         with open(self.config['checkpoint_file'], 'rb') as fp:
    #             model_checkpoint = pickle.load(fp)
    #         logging.debug('Loading model_checkpoint {}'.format(self.config['checkpoint_file']))

    #     self.model.load_state_dict(model_checkpoint)

    #     self.model.eval()

    #     total_labels = []
    #     total_preds = []
    #     with torch.no_grad():
    #         for step, (inputs, labels) in enumerate(self.dataloaders['test']):
    #             logging.debug('step {}'.format(step))
    #             inputs = inputs.to(self.config['device'])
    #             labels = labels.to(self.config['device'])

    #             outputs = self.model(inputs)
    #             _, preds = torch.max(outputs, 1)

    #             labels = labels.cpu().numpy()

    #             logits = outputs.cpu().numpy()
    #             preds = np.argmax(logits, axis=1)

    #             total_labels += labels.tolist()
    #             total_preds += preds.tolist()

    #         logging.debug('\nConfusionMatrix:\n{}'.format(confusion_matrix(y_true=total_labels, y_pred=total_preds)))

