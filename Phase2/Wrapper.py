import argparse
import glob
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
import imageio
import torch
import matplotlib.pyplot as plt

from DataLoader import *
from NeRFModel import *
from PositionEncoder import *
from render import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
np.random.seed(0)

def volumetricLoss(rgbMap, rgbMapGT):
    loss = torch.mean((rgbMap - rgbMapGT)**2)
    return loss

def render(rays, model, near, far, args, batchSize = 1024):
    rays_o, rays_d = rays
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    numRay = rays_o.shape[0]

    pts,viewdirs,z_vals = sample(rays_o,rays_d,near,far, args.n_sample)

    view_directions = viewdirs.view(numRay, 1, 3)
    view_directions = view_directions.expand([numRay, args.n_sample, 3])

    # posEnc = positionEncoder(pts, args.n_pos_freq).reshape(numRay*args.n_sample, -1)
    # dirEnc = positionEncoder(view_directions, args.n_dirc_freq).reshape(numRay*args.n_sample, -1)
    # raw = torch.cat([model(posEnc[i:i+batchSize], dirEnc[i:i+batchSize]) for i in range(0, posEnc.shape[0], batchSize)])

    outputs = []
    for i in range(0, pts.shape[0], batchSize):
        # print(pts.shape[0]// batchSize, i // batchSize)
        # print(torch.cuda.mem_get_info())
        torch.cuda.empty_cache()
        numRay = pts[i:i+batchSize, ...].shape[0]
        posEnc = positionEncoder(pts[i:i+batchSize, ...], args.n_pos_freq).reshape(numRay*args.n_sample, -1)
        dirEnc = positionEncoder(view_directions[i:i+batchSize, ...], args.n_dirc_freq).reshape(numRay*args.n_sample, -1)
        output = model(posEnc, dirEnc)
        # if args.mode=="test":
        #     output = output.detach().cpu()
        outputs.append(output)
        # posEnc.detach().cpu()
        # dirEnc.detach().cpu()
    raw = torch.cat(outputs)
    raw = raw.reshape((-1, args.n_sample, 4)).to(device)

    rgbMap = volumeRender(raw, z_vals, viewdirs)

    return rgbMap

def loadModel(model, args):
    startIter = 0
    files = glob.glob(args.checkpoint_path + '*.ckpt')
    latest_ckpt_file = max(files, key=os.path.getctime) if files else None
    print(files)

    if latest_ckpt_file and args.load_checkpoint:
        print(latest_ckpt_file)
        latest_ckpt = torch.load(latest_ckpt_file, map_location=torch.device(device))
        # startIter = latest_ckpt_file.replace(args.checkpoint_path,'').replace('model_','').replace('.ckpt','')
        # startIter = int(startIter)
        model.load_state_dict(latest_ckpt['model_state_dict'])
        print(f"Loaded latest checkpoint from {latest_ckpt_file} ....")
    else:
        print('New model initialized....')
    
    return startIter

def train(images, poses, hwf, K, near, far, args):
    # setup tensorboard
    writer = SummaryWriter(args.logs_path)

    H, W, focal = hwf
    poses = torch.Tensor(poses).to(device)

    # setup encoder
    dimPosition = 3 * 2 * args.n_pos_freq + 3
    dimDirection = 3 * 2 * args.n_dirc_freq + 3

    # create model
    model = NeRFmodel(dimPos = dimPosition, dimDirc=dimDirection).to(device)
    # init optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lrate, betas=(0.9, 0.999))
    # load checkpoint
    startIter = loadModel(model, args)

    # start iteration
    numImg = len(images)
    for iter in tqdm(range(startIter, args.max_iters)):
        # choose random image
        imgID = random.randint(0, numImg-1)
        imageGT = images[imgID]
        imageGT = torch.Tensor(imageGT).to(device)
        poseGT = poses[imgID, :3,:4]

        numRay = args.n_rays_batch
        rays_o, rays_d = cameraFrameToRays(H, W, K, torch.Tensor(poseGT))

        if iter < args.center_crop_iter:
            scale = 0.5
            dH = int(H//2 * scale)
            dW = int(W//2 * scale)
            coords = torch.stack(torch.meshgrid(torch.linspace(H//2-dH, H//2+dH-1, 2*dH), 
                                                torch.linspace(W//2-dW, W//2+dW-1, 2*dW)), -1)
        else:
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)
        coords = torch.reshape(coords, [-1,2])
        
        # choose random rays
        select_inds = np.random.choice(coords.shape[0], size=[numRay], replace=False)
        select_coords = coords[select_inds].long()
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]
        batch_rays = torch.stack([rays_o, rays_d], 0)
        rgbMapGT = imageGT[select_coords[:, 0], select_coords[:, 1]]

        model.train()
        rgbMap = render(batch_rays, model, near, far, args)
        lossIter = volumetricLoss(rgbMap, rgbMapGT)

        optimizer.zero_grad()
        lossIter.backward()
        optimizer.step()
        
        if iter % 10 == 0:
            print(f"iter:{iter}, loss_this_iter:{lossIter}\n")
            writer.add_scalar('LossEveryIter', lossIter, iter)
            writer.flush()

        # save checkpoint
        if iter % args.save_ckpt_iter == 0:
            print("Saved a checkpoint {}".format(iter))
            if not (os.path.isdir(args.checkpoint_path)):
                os.makedirs(args.checkpoint_path)
            
            checkpoint_save_name =  args.checkpoint_path + os.sep + 'model_' + str(iter) + '.ckpt'
            torch.save({'iter': iter,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': lossIter}, checkpoint_save_name)

    print("Training is done")

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def test(hwf, K, near, far, args):
    render_poses = getRenderPose()
    render_poses = torch.Tensor(render_poses).to(device)
    H, W, focal = hwf

    dimPosition = 3 * 2 * args.n_pos_freq + 3
    dimDirection = 3 * 2 * args.n_dirc_freq + 3
    model = NeRFmodel(dimPos = dimPosition, dimDirc=dimDirection).to(device)
    loadModel(model, args)
    model.eval()

    if not (os.path.isdir(args.images_path)):
        os.makedirs(args.images_path)

    images = []
    for i, pose in enumerate(render_poses):
        rays_o, rays_d = cameraFrameToRays(H, W, K, torch.Tensor(pose))
        batch_rays = torch.stack([rays_o, rays_d], 0)
        with torch.no_grad():
            rgbMap = render(batch_rays, model, near, far, args)
        image = rgbMap.reshape((H, W, 3))
        image = image.detach().cpu().numpy()
        image8 = to8b(image)
        filename = "{}view_{}.png".format(args.images_path, i)
        imageio.imwrite(filename, image8)
        images.append(image8)
        print("Saved image ", i)
    imageio.mimsave(args.images_path+'lego.gif', images, fps=10)


def main(args):
    # load data
    print("Loading data...")
    images, poses, hwf, K, near, far, split = loadDataset(args.data_path, testskip=args.testskip)
    i_train, i_val, i_test = split
    images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])

    if args.mode == 'train':
        print("Start training")
        train(images[i_train], poses[i_train], hwf, K, near, far, args)
    elif args.mode == 'test':
        print("Start testing")
        args.load_checkpoint = True
        test(hwf, K, near, far, args)

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default="./Phase2/data/lego/",help="dataset path")
    parser.add_argument('--mode',default='train',help="train/test/val")
    parser.add_argument('--lrate',default=5e-4,help="training learning rate")
    parser.add_argument('--n_pos_freq',default=10,help="number of positional encoding frequencies for position")
    parser.add_argument('--n_dirc_freq',default=4,help="number of positional encoding frequencies for viewing direction")
    parser.add_argument('--n_rays_batch',default=32*32*4,help="number of rays per batch")
    parser.add_argument('--n_sample',default=400,help="number of sample per ray")
    parser.add_argument('--max_iters',default=10000,help="number of max iterations for training")
    parser.add_argument('--logs_path',default="./logs/",help="logs path")
    parser.add_argument('--checkpoint_path',default="./Phase2/example_checkpoint/",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=True,help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter',default=1000,help="num of iteration to save checkpoint")
    parser.add_argument('--center_crop_iter',default=500,help="center crop image for training before this iteration")
    parser.add_argument('--images_path', default="./image/",help="folder to store images")
    parser.add_argument('--testskip', default=8,help="downsample test images")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main(args)