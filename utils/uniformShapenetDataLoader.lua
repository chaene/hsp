local matio = require('matio')

math.randomseed( os.time() )

uniform3DShapeNetDataSet = {}
uniform3DShapeNetDataSet.resolutionStr = "32_smooth"
uniform3DShapeNetDataSet.numViews = 20

uniform3DShapeNetDataSet.testDataFiles = {}
uniform3DShapeNetDataSet.trainDataFiles = {}
uniform3DShapeNetDataSet.valDataFiles = {}

uniform3DShapeNetDataSet.lastTrainInd = 0
uniform3DShapeNetDataSet.lastValInd = 0
uniform3DShapeNetDataSet.lastTestInd = 0

uniform3DShapeNetDataSet.useColor = false

uniform3DShapeNetDataSet.hard = true

function uniform3DShapeNetDataSet:setPath(path)
  self.path = path
end

function uniform3DShapeNetDataSet:setSoft()
  uniform3DShapeNetDataSet.hard = false
end

function uniform3DShapeNetDataSet:setHard()
  uniform3DShapeNetDataSet.hard = true
end

function uniform3DShapeNetDataSet:setResolution(resolutionStr)
  self.resolutionStr = resolutionStr
end


function uniform3DShapeNetDataSet:addTestDataFromClass(class)
    io.input(self.path .. "/test_" .. class .. ".txt")
    for line in io.lines() do
      table.insert(self.testDataFiles, class .. "/" .. line)
    end
end

function uniform3DShapeNetDataSet:addValDataFromClass(class)
    io.input(self.path .. "/val_" .. class .. ".txt")
    for line in io.lines() do
        table.insert(self.valDataFiles, class .. "/" .. line)
    end
end

function uniform3DShapeNetDataSet:addTrainDataFromClass(class)
    io.input(self.path .. "/train_" .. class .. ".txt")
    for line in io.lines() do
        table.insert(self.trainDataFiles, class .. "/" .. line)
    end
end

function uniform3DShapeNetDataSet:shuffleTrainDataFileNames()
  
  rP = torch.randperm(#self.trainDataFiles)
  
  trainDataFilesPerm = {}
  for i = 1,#self.trainDataFiles do
    trainDataFilesPerm[i] = self.trainDataFiles[rP[i]]
  end
  
  self.trainDataFiles = trainDataFilesPerm
end

function uniform3DShapeNetDataSet:loadSample(fileNames, modelID, viewID)
  
  local obs 
  local voxels = {}
  local modelName = string.sub(fileNames[modelID],1,-5)
    
  if self.useColor then
    local fileName = self.path .. "/blenderRenderPreprocess/" .. string.sub(fileNames[modelID] ,1, -5) .. "/" .. "render_" .. viewID-1 .. ".png"
	
    -- white background
    local img = image.load(fileName)
    img = image.scale(img, 128, 128)
    local alphaMask = img[4]:repeatTensor(3,1,1)
    img = torch.cmul(img:narrow(1,1,3),alphaMask) + 1 - alphaMask
    --local noise = torch.Tensor(3,128,128)
    --noise:uniform()
    obs = img --+noise*0.01
  else
    local fileName = self.path .. "/blenderRenderPreprocess/"  .. string.sub(fileNames[modelID] ,1, -5) .. "/" .. "depth_" .. viewID-1 .. ".png"
    local dm = image.load(fileName)
    dm = image.scale(dm, 128, 128, 'simple')
    obs = dm:add(-1)
  end
  
  local volume = matio.load(self.path .. "/modelVoxels" .. self.resolutionStr 
			     .."/" .. string.sub(fileNames[modelID],1,-5) .. ".mat", {'Volume'})
  
  voxels = volume.Volume
  
  if self.hard then
    voxels[torch.gt(voxels, 0)] = 1
  end
			      
  return obs, voxels, modelName
end

function uniform3DShapeNetDataSet:setColor()
  self.useColor = true
end

function uniform3DShapeNetDataSet:getNextSample(fileNames, last, randomizeElement, randomizeView)
  
  local ind = 0
  if (randomizeElement) then
    ind = math.random(fileNames)
  else
    last = last + 1
    if last > #fileNames then
      last = 1
    end
    ind = last
  end
  
  local view = 1
  if (randomizeView) then
    view = math.random(uniform3DShapeNetDataSet.numViews)
  end
  
  local obs, voxels, modelName = self:loadSample(fileNames, ind, view)
  
  return ind, obs, voxels, modelName
end

function uniform3DShapeNetDataSet:getNextTrainExample(randomizeElement, randomizeView)
  
  local ind, obs, voxels, modelName = self:getNextSample(self.trainDataFiles, self.lastTrainInd, randomizeElement, randomizeView)
  --if (ind > 100) then
  --  ind = 0
  --end
  self.lastTrainInd = ind

  
  return obs, voxels, modelName
end


function uniform3DShapeNetDataSet:getNextValExample(randomize, randomizeView)
  
  local ind, obs, voxels, modelName = self:getNextSample(self.valDataFiles, self.lastValInd, randomizeElement, randomizeView)
  self.lastValInd = ind
  
  return obs,voxels, modelName
end

-- useful to validate on subset
function uniform3DShapeNetDataSet:skipNextValExample()
  self.lastValInd = self.lastValInd + 1
    if self.lastValInd > #self.valDataFiles then
      self.lastValInd = 1
    end
    ind = self.lastValInd
end

function uniform3DShapeNetDataSet:getNextTestExample(randomize, randomizeView)
  
  local ind, obs, voxels, modelName = self:getNextSample(self.testDataFiles, self.testValInd, randomizeElement, randomizeView)
  self.lastTestInd = ind
  
  return obs, voxels, modelName
end









