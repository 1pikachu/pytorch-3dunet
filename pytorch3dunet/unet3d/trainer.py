import os
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from pytorch3dunet.datasets.utils import get_train_loaders
from pytorch3dunet.unet3d.losses import get_loss_criterion
from pytorch3dunet.unet3d.metrics import get_evaluation_metric
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.utils import get_logger, get_tensorboard_formatter, create_optimizer, \
    create_lr_scheduler, get_number_of_learnable_parameters
from . import utils

logger = get_logger('UNet3DTrainer')


def create_trainer(config):
    # Create the model
    model = get_model(config['model'])
    # use DataParallel if more than 1 GPU available
    device = config['device']
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for training')

    # put the model on GPUs
    logger.info(f"Sending the model to '{config['device']}'")
    model = model.to(device)

    if config['channels_last'] and device != "xpu":
        try:
            model = model.to(memory_format=torch.channels_last_3d)
            print("--- use NDHWC format")
        except RuntimeError as e:
            print("---- use normal format")
            print("failed to enable NHWC: ", e)

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create loss criterion
    loss_criterion = get_loss_criterion(config)
    # Create evaluation metric
    eval_criterion = get_evaluation_metric(config)

    # Create data loaders
    loaders = get_train_loaders(config)

    # Create the optimizer
    optimizer = create_optimizer(config['optimizer'], model)
    if config['device'].__str__() == "xpu":
        datatype = torch.float16 if config['precision'] == "float16" else torch.bfloat16 if config['precision'] == "bfloat16" else torch.float
        model, optimizer = torch.xpu.optimize(model=model, optimizer=optimizer, dtype=datatype)
        print("---- enable xpu optimize")

    # Create learning rate adjustment strategy
    lr_scheduler = create_lr_scheduler(config.get('lr_scheduler', None), optimizer)

    trainer_config = config['trainer']
    # Create tensorboard formatter
    tensorboard_formatter = get_tensorboard_formatter(trainer_config.pop('tensorboard_formatter', None))
    # Create trainer
    resume = trainer_config.pop('resume', None)
    pre_trained = trainer_config.pop('pre_trained', None)

    return UNet3DTrainer(model=model,
                         optimizer=optimizer,
                         lr_scheduler=lr_scheduler,
                         loss_criterion=loss_criterion,
                         eval_criterion=eval_criterion,
                         tensorboard_formatter=tensorboard_formatter,
                         device=config['device'],
                         loaders=loaders,
                         resume=resume,
                         pre_trained=pre_trained,
                         config=config,
                         **trainer_config)


class UNet3DTrainer:
    """3D UNet trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
        tensorboard_formatter (callable): converts a given batch of input/output/target image to a series of images
            that can be displayed in tensorboard
        skip_train_validation (bool): if True eval_criterion is not evaluated on the training set (used mostly when
            evaluation is expensive)
    """

    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 eval_criterion, device, loaders, checkpoint_dir,
                 max_num_epochs, max_num_iterations,
                 validate_after_iters=200, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=True,
                 tensorboard_formatter=None, skip_train_validation=False,
                 resume=None, pre_trained=None, config=None, **kwargs):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.config = config

        logger.info(model)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        # initialize the best_eval_score
        if eval_score_higher_is_better:
            self.best_eval_score = float('-inf')
        else:
            self.best_eval_score = float('+inf')

        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))

        assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter

        self.num_iterations = 0
        self.num_epochs = 0
        self.skip_train_validation = skip_train_validation

        if resume is not None:
            logger.info(f"Loading checkpoint '{resume}'...")
            state = utils.load_checkpoint(resume, self.model, self.optimizer)
            logger.info(
                f"Checkpoint loaded from '{resume}'. Epoch: {state['num_epochs']}.  Iteration: {state['num_iterations']}. "
                f"Best val score: {state['best_eval_score']}."
            )
            self.best_eval_score = state['best_eval_score']
            self.num_iterations = state['num_iterations']
            self.num_epochs = state['num_epochs']
            self.checkpoint_dir = os.path.split(resume)[0]
        elif pre_trained is not None:
            logger.info(f"Logging pre-trained model from '{pre_trained}'...")
            utils.load_checkpoint(pre_trained, self.model, None)
            if 'checkpoint_dir' not in kwargs:
                self.checkpoint_dir = os.path.split(pre_trained)[0]

    def fit(self):
        for _ in range(self.num_epochs, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train()

            if should_terminate:
                logger.info('Stopping criterion is satisfied. Finishing training')
                return

            self.num_epochs += 1
        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def train(self):
        """Trains the model for 1 epoch.

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = utils.RunningAverage()
        train_eval_scores = utils.RunningAverage()

        # sets the model in training mode
        self.model.train()

        total_time = 0.0
        total_count = 0
        profile_len = min(len(self.loaders['train']), self.config['num_iter']) // 2
        last_iter = min(len(self.loaders['train']), self.config['num_iter']) - 1
        datatype = torch.float16 if self.config['precision'] == "float16" else torch.bfloat16 if self.config['precision'] == "bfloat16" else torch.float
        if self.config['profile'] and self.config['device_str'] == "xpu":
            for t in self.loaders['train']:
                logger.info(f'Training iteration [{self.num_iterations}/{self.max_num_iterations}]. '
                            f'Epoch [{self.num_epochs}/{self.max_num_epochs - 1}]')

                start_time = time.time()
                # to device
                input, target, weight = self._split_training_batch(t)
                with torch.autograd.profiler_legacy.profile(enabled=True, use_xpu=True, record_shapes=False) as prof:
                    with torch.xpu.amp.autocast(enabled=True, dtype=datatype):
                        output, loss = self._forward_pass(input, target, weight)

                train_losses.update(loss.item(), self._batch_size(input))

                # compute gradients and update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                output = output.cpu()
                loss = loss.cpu()
                torch.xpu.synchronize()
                duration = time.time() - start_time
                print("iteration:{}, training time: {} sec.".format(self.num_iterations, duration))
                if self.num_iterations >= self.config['num_warmup'] and self.num_iterations < last_iter:
                    print("the iteration-{} has been calcute perf".format(self.num_iterations))
                    total_time += duration
                    total_count += 1
                if args.profile and self.num_iterations == profile_len:
                    import pathlib
                    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                    if not os.path.exists(timeline_dir):
                        try:
                            os.makedirs(timeline_dir)
                        except:
                            pass
                    torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"),
                        timeline_dir+'profile.pt')
                    torch.save(prof.key_averages(group_by_input_shape=True).table(),
                        timeline_dir+'profile_detail.pt')
                    torch.save(prof.table(sort_by="id", row_limit=100000),
                        timeline_dir+'profile_detail_withId.pt')
                    prof.export_chrome_trace(timeline_dir+"trace.json")
                if self.num_iterations >= self.config['num_iter']:
                    break

                self.num_iterations += 1
        
        elif self.config['profile'] and self.config['device_str'] != "xpu":
            if self.config['device_str'] == "cuda":
                profile_act = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            else:
                profile_act = [torch.profiler.ProfilerActivity.CPU]
            with torch.profiler.profile(
                activities=profile_act,
                record_shapes=True,
                schedule=torch.profiler.schedule(
                    wait=profile_len,
                    warmup=2,
                    active=1,
                ),
                on_trace_ready=self.trace_handler,
            ) as p:
                for t in self.loaders['train']:
                    logger.info(f'Training iteration [{self.num_iterations}/{self.max_num_iterations}]. '
                                f'Epoch [{self.num_epochs}/{self.max_num_epochs - 1}]')

                    start_time = time.time()
                    # to device
                    input, target, weight = self._split_training_batch(t)

                    cl_start_time = time.time()
                    if self.config['channels_last'] and self.config['device_str'] != "xpu":
                        try:
                            input = input.to(memory_format=torch.channels_last_3d)
                            print("---input use NDHWC format")
                        except RuntimeError as e:
                            print("----input use normal format")
                            print("failed to enable NHWC: ", e)
                    cl_duration = time.time() - cl_start_time

                    if self.config['device_str'] == "cuda":
                        with torch.cuda.amp.autocast(enabled=True, dtype=datatype):
                            output, loss = self._forward_pass(input, target, weight)
                    else:
                        with torch.cpu.amp.autocast(enabled=True, dtype=datatype):
                            output, loss = self._forward_pass(input, target, weight)

                    train_losses.update(loss.item(), self._batch_size(input))

                    # compute gradients and update parameters
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    output = output.cpu()
                    loss = loss.cpu()
                    if self.config['device_str'] == "cuda":
                        torch.cuda.synchronize()
                    duration = time.time() - start_time - cl_duration
                    p.step()
                    print("iteration:{}, training time: {} sec.".format(self.num_iterations, duration))
                    if self.num_iterations >= self.config['num_warmup'] and self.num_iterations < last_iter:
                        print("the iteration-{} has been calcute perf".format(self.num_iterations))
                        total_time += duration
                        total_count += 1
                    if self.num_iterations >= self.config['num_iter']:
                        break

                    self.num_iterations += 1

        else:
            for t in self.loaders['train']:
                logger.info(f'Training iteration [{self.num_iterations}/{self.max_num_iterations}]. '
                            f'Epoch [{self.num_epochs}/{self.max_num_epochs - 1}]')

                start_time = time.time()
                # to device
                input, target, weight = self._split_training_batch(t)

                cl_start_time = time.time()
                if self.config['channels_last'] and self.config['device_str'] != "xpu":
                    try:
                        input = input.to(memory_format=torch.channels_last_3d)
                        print("---input use NDHWC format")
                    except RuntimeError as e:
                        print("----input use normal format")
                        print("failed to enable NHWC: ", e)
                cl_duration = time.time() - cl_start_time

                if self.config['device_str'] == "cuda":
                    with torch.cuda.amp.autocast(enabled=True, dtype=datatype):
                        output, loss = self._forward_pass(input, target, weight)
                elif self.config['device_str'] == "xpu":
                    with torch.xpu.amp.autocast(enabled=True, dtype=datatype):
                        output, loss = self._forward_pass(input, target, weight)
                else:
                    with torch.cpu.amp.autocast(enabled=True, dtype=datatype):
                        output, loss = self._forward_pass(input, target, weight)

                train_losses.update(loss.item(), self._batch_size(input))

                # compute gradients and update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                output = output.cpu()
                loss = loss.cpu()
                if self.config['device_str'] == "cuda":
                    torch.cuda.synchronize()
                elif self.config['device_str'] == "xpu":
                    torch.xpu.synchronize()
                duration = time.time() - start_time - cl_duration
                print("iteration:{}, training time: {} sec.".format(self.num_iterations, duration))
                if self.num_iterations >= self.config['num_warmup'] and self.num_iterations < last_iter:
                    print("the iteration-{} has been calcute perf".format(self.num_iterations))
                    total_time += duration
                    total_count += 1
                if self.num_iterations >= self.config['num_iter']:
                    break

                self.num_iterations += 1

        batch_size = self.config['loaders']['batch_size']
        avg_time = total_time / total_count
        latency = avg_time / batch_size * 1000
        perf = batch_size / avg_time
        print("total time:{}, total count:{}".format(total_time, total_count))
        print('%d epoch training latency: %6.2f ms'%(0, latency))
        print('%d epoch training Throughput: %6.2f fps'%(0, perf))
        return True

    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            logger.info(f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

        min_lr = 1e-6
        lr = self.optimizer.param_groups[0]['lr']
        if lr < min_lr:
            logger.info(f'Learning rate below the minimum {min_lr}.')
            return True

        return False

    def validate(self):
        logger.info('Validating...')

        val_losses = utils.RunningAverage()
        val_scores = utils.RunningAverage()

        with torch.no_grad():
            for i, t in enumerate(self.loaders['val']):
                logger.info(f'Validation iteration {i}')

                input, target, weight = self._split_training_batch(t)

                output, loss = self._forward_pass(input, target, weight)
                val_losses.update(loss.item(), self._batch_size(input))

                if i % 100 == 0:
                    self._log_images(input, target, output, 'val_')

                eval_score = self.eval_criterion(output, target)
                val_scores.update(eval_score.item(), self._batch_size(input))

                if self.validate_iters is not None and self.validate_iters <= i:
                    # stop validation
                    break

            self._log_stats('val', val_losses.avg, val_scores.avg)
            logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
            return val_scores.avg

    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                return input.to(self.device)

        t = _move_to_device(t)
        weight = None
        if len(t) == 2:
            input, target = t
        else:
            input, target, weight = t
        return input, target, weight

    def _forward_pass(self, input, target, weight=None):
        # forward pass
        output = self.model(input)

        # compute the loss
        if weight is None:
            loss = self.loss_criterion(output, target)
        else:
            loss = self.loss_criterion(output, target, weight)

        return output, loss

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        last_file_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pytorch')
        logger.info(f"Saving checkpoint to '{last_file_path}'")

        utils.save_checkpoint({
            'num_epochs': self.num_epochs + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': state_dict,
            'best_eval_score': self.best_eval_score,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, is_best, checkpoint_dir=self.checkpoint_dir)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction, prefix=''):
        if self.model.training:
            if isinstance(self.model, nn.DataParallel):
                net = self.model.module
            else:
                net = self.model

            if net.final_activation is not None:
                prediction = net.final_activation(prediction)

        inputs_map = {
            'inputs': input,
            'targets': target,
            'predictions': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(prefix + tag, image, self.num_iterations)

    def trace_handler(self, p):
        output = p.key_averages().table(sort_by="self_cpu_time_total")
        print(output)
        import pathlib
        timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
        if not os.path.exists(timeline_dir):
            try:
                os.makedirs(timeline_dir)
            except:
                pass
        timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + \
                '-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
        p.export_chrome_trace(timeline_file)

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
