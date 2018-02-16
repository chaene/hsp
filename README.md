## Installation


Install [torch](http://torch.ch/)

Download [CImg](http://cimg.eu/) and place it in the torch-hsp subfolder. The file "CImg.h" needs to be in the path "torch-hsp/CImg/".

Install the torch package torch-hsp by running "luarocks make hsp-1.0-0.rockspec" in the torch-hsp folder



## Running Demo


A demo script is included which reconstructs a single image and outputs a mesh as obj file. It needs as input the pretrained network provided [here](https://drive.google.com/file/d/1it00XjWc7PnKAwVhPEtl2V96g3RPbi2V/view?usp=sharing).


th hspDemo.lua &lt;GPU ID&gt; &lt;Trained Network File Name&gt; &lt;Input Image File Name&gt;


## Training Network


Example parameter files are provided [here](https://drive.google.com/file/d/1it00XjWc7PnKAwVhPEtl2V96g3RPbi2V/view?usp=sharing).

The data is provided [here](https://drive.google.com/file/d/1xtJz5CEEPgYOtWP6Dr6nUWbUXPDMswh0/view?usp=sharing).

To train a network the paths to the shapenet dataset and the output folder in the "parameters.lua" file need to be adjusted first.

th trainNetworkHierarchical.lua &lt;GPU ID&gt; &lt;Parameter File Name&gt;
