from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='train', help='train,test')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        self.parser.add_argument('--g_lr', type=float, default=2e-4, help='initial learning rate for adam')
        self.parser.add_argument('--d_lr', type=float, default=2e-4, help='initial learning rate for adam')
        self.parser.add_argument('--lambda_A', type=float, default=10.0)
        self.parser.add_argument('--lambda_B', type=float, default=10.0)
        self.parser.add_argument('--lambda_idt', type=float, default=0.5)
        self.parser.add_argument('--lambda_his_lip', type=float, default=1.0)
        self.parser.add_argument('--lambda_his_skin', type=float, default=0.1)
        self.parser.add_argument('--lambda_his_eye', type=float, default=1.0)
        self.parser.add_argument('--lambda_vgg', type=float, default=5e-3)
        self.parser.add_argument('--num_epochs', type=int, default=100)
        self.parser.add_argument('--epochs_decay', type=int, default=0)
        self.parser.add_argument('--g_step', type=int, default=1)
        self.parser.add_argument('--log_step', type=int, default=8)
        self.parser.add_argument('--save_step', type=int, default=2048)
        self.parser.add_argument('--snapshot_path', type=str, default='./checkpoints/')
        self.parser.add_argument('--save_path', type=str, default='./results/')
        self.parser.add_argument('--snapshot_step', type=int, default=10)
        self.parser.add_argument('--perceptual_layers', type=int, default=3, help='index of vgg layer.')
        self.parser.add_argument('--partial', action='store_true', default=False, help='Partial Transfer')
        self.parser.add_argument('--interpolation', action='store_true', default=False, help='Interpolation')
