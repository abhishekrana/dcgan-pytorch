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
                # logging.info('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                logging.info('[{}/{}][{}/{}] Loss_D: {:.4} Loss_G: {:.4} D(x): {:.4} D(G(z)): {:.4} / {:.4}'
                      .format(epoch, self.epochs, i, len(self.dataloaders['train']),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                
            if epoch % 10 == 0:
                vutils.save_image(real_cpu,
                        '%s/real_samples.png' % self.config['output_images_path'],
                        normalize=True)
                fake = self.model.netG(self.fixed_noise)
                vutils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % (self.config['output_images_path'], epoch),
                    normalize=True)

            # Save checkpoints
            if epoch % 50 == 0:
                torch.save(self.model.netG.state_dict(), '%s/netG_epoch_%d.pth' % (self.config['checkpoint_path'], epoch))
                torch.save(self.model.netD.state_dict(), '%s/netD_epoch_%d.pth' % (self.config['checkpoint_path'], epoch))


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

