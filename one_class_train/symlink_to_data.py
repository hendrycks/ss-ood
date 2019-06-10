import os, errno

def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except:
        os.remove(link_name)
        os.symlink(target, link_name)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

pairs = [
('acorn', 'n12267677'),
('airliner', 'n02690373'),
('ambulance', 'n02701002'),
('american_alligator', 'n01698640'),
('banjo', 'n02787622'),
('barn', 'n02793495'),
('bikini', 'n02837789'),
('digital_clock', 'n03196217'),
('dragonfly', 'n02268443'),
('dumbbell', 'n03255030'),
('forklift', 'n03384352'),
('goblet', 'n03443371'),
('grand_piano', 'n03452741'),
('hotdog', 'n07697537'),
('hourglass', 'n03544143'),
('manhole_cover', 'n03717622'),
('mosque', 'n03788195'),
('nail', 'n03804744'),
('parking_meter', 'n03891332'),
('pillow', 'n03938244'),
('revolver', 'n04086273'),
('rotary_dial_telephone', 'n03187595'),
('schooner', 'n04147183'),
('snowmobile', 'n04252077'),
('soccer_ball', 'n04254680'),
('stingray', 'n01498041'),
('strawberry', 'n07745940'),
('tank', 'n04389033'),
('toaster', 'n04442312'),
('volcano', 'n09472597')
]

for pair in pairs:
    pth = pair[0]   # os.path.join('/home/hendrycks/datasets/imagenet/one_class_train', pair[0])
    mkdir(pth)
    symlink_force(os.path.join('/share/data/vision-greg/ImageNet/clsloc/images/train/', pair[1]), os.path.join(pth, 'images'))
