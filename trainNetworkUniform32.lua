require('nn')
require('cudnn')
require('cutorch')
require('cunn')
require 'image'
require 'torch'
require 'optim'
require 'socket'
require 'lfs'

local hsp = require 'hsp'

math.randomseed(os.time())

dofile('networks/networkUniform32.lua')

dofile('utils/uniformShapenetDataLoader.lua')

local networkParameters = {}
local trainingParams = {}

local arg = arg

networkParameters, trainingParams = dofile(arg[2])
cutorch.setDevice(arg[1])

local startIteration = 1
local net
local optimState = {}

if not trainingParams.resumeIteration then
  if (networkParameters.useColor) then
    net = uniformNetwork.generateNewNetwork(3,networkParameters.bottleNeckSize):cuda()
  else
    net = uniformNetwork.generateNewNetwork(1,networkParameters.bottleNeckSize):cuda()
  end
else
  net, optimState = uniformNetwork.loadSnapshot(trainingParams.outputFolder .. "/snapshot/", trainingParams.resumeIteration, true)
  startIteration = optimState.iter + 1
  net:training()
end


valLogger = optim.Logger(trainingParams.outputFolder .. 'training_valLoss_' .. startIteration .. '.log')
valLogger:setNames{'Iteration', 'L', 'time'}

trainLogger = optim.Logger(trainingParams.outputFolder .. 'training_trainLoss_' .. startIteration .. '.log')
trainLogger:setNames{'Iteration', 'L', 'time'}

local optimParams = trainingParams.config

-- set up the data set
uniform3DShapeNetDataSet:setResolution(trainingParams.resolutionStr)

if networkParameters.useColor then
  uniform3DShapeNetDataSet:setColor()
end

if networkParameters.hard then
  uniform3DShapeNetDataSet:setHard()
else
  uniform3DShapeNetDataSet:setSoft()
end

uniform3DShapeNetDataSet:setPath(trainingParams.datasetFolder)
for c=1,#trainingParams.valClasses do
  uniform3DShapeNetDataSet:addValDataFromClass(trainingParams.valClasses[c])
end

for c=1,#trainingParams.trainClasses do
  uniform3DShapeNetDataSet:addTrainDataFromClass(trainingParams.trainClasses[c])
end

uniform3DShapeNetDataSet:shuffleTrainDataFileNames()

-- generate snapshot folder
local snapshotFolderExists = lfs.attributes(trainingParams.outputFolder .. "/snapshot/",'modification')
if (snapshotFolderExists == nil) then
  lfs.mkdir(trainingParams.outputFolder .. "/snapshot/")
end


-- sliding window training loss
local trainingLossBufferSize = 200
local trainingLossBuffer = torch.Tensor(trainingLossBufferSize):zero()
local trainingLossBufferPointer = 1
local loss = 0

local totalTrainingTime = 0

cutorch.synchronize()
local start = socket.gettime()

-- loss layers
local bceCrit = nn.BCECriterion():cuda()


-- allocate structures for the input to the network
local batchInput
if (networkParameters.useColor) then
  batchInput = torch.Tensor(trainingParams.batchSize, 3, 128, 128):cuda()
else
  batchInput = torch.Tensor(trainingParams.batchSize, 1, 128, 128):cuda()
end
local batchVoxels = torch.Tensor(trainingParams.batchSize, 1, 32, 32, 32):cuda()

local params, gradParams = net:getParameters()

for i = startIteration, 1000000 do
  
  local function evalGradients(params)
    gradParams:zero()
    
    -- run first stage 
    local result = net:forward(batchInput)
    loss = bceCrit:forward(result, batchVoxels)
    

    
    local gradLosss = bceCrit:backward(result, batchVoxels)
    net:backward(batchInput, gradLosss)
    
    return loss, gradParams
  end
  
  -- load a new batch
  for b=1,trainingParams.batchSize do
    local obs, voxels, modelName = uniform3DShapeNetDataSet:getNextTrainExample(false, true)
    batchInput[{b,{},{},{}}] = obs:cuda()
    batchVoxels[{b,{},{},{},{}}] = voxels:cuda()
  end
  
  -- do the optimization step
  optim.adam(evalGradients, params, optimParams, optimState)
  
  trainingLossBuffer[trainingLossBufferPointer] = loss
  trainingLossBufferPointer = trainingLossBufferPointer + 1
  if (trainingLossBufferPointer > trainingLossBufferSize) then
    trainingLossBufferPointer = 1
  end
   
  -- output loss every 5 iterations
  if i % 5 == 0 then
    local slidingWindowLoss = torch.sum(trainingLossBuffer)/math.min(trainingLossBufferSize,i-startIteration+1)
    print("iteration: " .. i .. ", sliding window training loss : ")
    print(slidingWindowLoss)
    
    local curEnd = socket.gettime()
      
    trainLogger:add{i,slidingWindowLoss, curEnd - start}
  end
  
  if (i % trainingParams.validationInterval == 0) then
    uniformNetwork.saveSnapshot(i, trainingParams.outputFolder .. "/snapshot/", net, optimState)
    
    net:evaluate()
    
    local valBatchSize = 1
    
    local valLoss = 0
    local numValidatedElements = 0
    
    for j=1,#uniform3DShapeNetDataSet.valDataFiles do
      if ((j-1)%trainingParams.validationSubsampling == 0) then
        local obs, voxels, modelName = uniform3DShapeNetDataSet:getNextValExample(false, false)
        
	if networkParameters.useColor then
	  input = obs:cuda():view(1,3,128,128)
	else
	  input = obs:cuda():view(1,1,128,128)
	end
	
        voxels = voxels:cuda()
        
        print("Running validation model " .. j .. " : " .. modelName)

        -- run first stage 
        result = net:forward(input)
        loss = bceCrit:forward(result, voxels)
	
	
	--hsp.saveMeshAsObj(result[{1,1,{},{},{}}]:double(),0.25, trainingParams.outputFolder .. "/objs/" .. modelName .. ".obj")

        valLoss = valLoss + loss
        numValidatedElements = numValidatedElements + 1
        
      else
        uniform3DShapeNetDataSet:skipNextValExample()
      end
    end
    
    valLoss = valLoss / numValidatedElements
    
    local curEnd = socket.gettime()
    valLogger:add{i, valLoss, curEnd - start}
    
    
    net:training()
    
  end
end
 
