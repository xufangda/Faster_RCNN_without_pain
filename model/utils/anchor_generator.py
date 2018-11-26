import numpy as np

def anchor_generator(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    
    px = base_size/2.0
    py = base_size/2.0
    anchor_base=np.zeros((len(ratios)*len(anchor_scales),4))
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            w = base_size*anchor_scales[j]*np.sqrt(ratios[i])
            h = base_size*anchor_scales[j]*np.sqrt(1./ratios[i])

            index = i * len(ratios) + j 
            anchor_base[index, 0] = px - 0.5 * w
            anchor_base[index, 1] = py - 0.5 * h
            anchor_base[index, 2] = px + 0.5 * w
            anchor_base[index, 3] = py + 0.5 * h

    return anchor_base

if __name__ == "__main__":
    t=anchor_generator()
    print(t)    