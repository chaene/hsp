require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'optim'
require 'cudnn'
dofile('utils/instanceNormalization.lua')
require 'lfs'
require 'image'

local hsp = require 'hsp'

function split(inputstr, sep)
  if sep == nil then
    sep = "%s"
  end
  local t={} ; i=1
  for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
    t[i] = str
    i = i + 1
  end
  return t
end

networkParameters, evalParams  = dofile(arg[2])
cutorch.setDevice(arg[1])

local splitStr = evalParams.split
local objStr = ""
if evalParams.saveObj then
  objStr = "_obj"
end

local metric = evalParams.metric

dofile('utils/hierarchicalShapenetDataLoader.lua')
hierarchical3DShapeNetDataSet:setResolution(evalParams.resolutionStr)
if networkParameters.useColor then
  hierarchical3DShapeNetDataSet:setColor()
end
hierarchical3DShapeNetDataSet:setPath(evalParams.datasetFolder)
for c=1,#evalParams.classes do
  if splitStr == "val" then
    hierarchical3DShapeNetDataSet:addValDataFromClass(evalParams.classes[c])
  elseif splitStr == "test" then
    hierarchical3DShapeNetDataSet:addTestDataFromClass(evalParams.classes[c])
  end
end

local peakPerformance = 0
local peakPerformanceIter = 0
local peakPerformanceThresh = 0
if metric == "CD" then
  peakPerformance = 10 -- value which is bigger than biggest CD
end

local thresholds = evalParams.thresholds
local numLevels = networkParameters.numLevels

local AvGIoUAvgFile = assert(io.open(evalParams.outputFolder .. "/evaluation_" .. splitStr .. objStr .."_AvG" .. metric .. "_Avg_" .. evalParams.snapshotStartIter .. ".txt", "w"))
AvGIoUAvgFile:write("iter")
for t=1,#thresholds do
  AvGIoUAvgFile:write(", " .. thresholds[t])
end
AvGIoUAvgFile:write("\n")

local AvGIoUFiles = {}
for c=1,#evalParams.classes do
  AvGIoUFiles[evalParams.classes[c]] = assert(io.open(evalParams.outputFolder .. "/evaluation_" .. splitStr .. objStr .. "_AvG" .. metric .. "_" .. evalParams.classes[c] .. "_" .. evalParams.snapshotStartIter .. ".txt", "w"))
  AvGIoUFiles[evalParams.classes[c]]:write("iter")
  for t=1,#thresholds do
    AvGIoUFiles[evalParams.classes[c]]:write(", " .. thresholds[t])
  end
  AvGIoUFiles[evalParams.classes[c]]:write("\n")
end

--generate obj folders if objs output is written
if (evalParams.saveObj == true) then
  local objFolderExists = lfs.attributes(evalParams.outputFolder .. "/objs_" ..splitStr .. "/",'modification')
  if (objFolderExists == nil) then
    lfs.mkdir(evalParams.outputFolder .. "/objs_" .. splitStr .. "/",'modification')
  end
  for c=1,#evalParams.classes do
    local classFolderExists = lfs.attributes(evalParams.outputFolder .. "/objs_" .. splitStr .. "/" .. evalParams.classes[c], 'modification')
    if (classFolderExists == nil) then
      lfs.mkdir(evalParams.outputFolder .. "/objs_" .. splitStr .. "/" .. evalParams.classes[c])
    end
  end
end

for iter=evalParams.snapshotStartIter,evalParams.snapshotEndIter,evalParams.snapshotInterval do
  
  local totalIoU = {}
  local numEvaluatedElements = {}
  for c=1,#evalParams.classes do
    totalIoU[evalParams.classes[c]] = {}
    numEvaluatedElements[evalParams.classes[c]] = 0
    for t=1,#thresholds do
      totalIoU[evalParams.classes[c]][t] = 0
    end
  end
  
  
  if not evalParams.loadCache then
    net = torch.load(evalParams.snapshotPath .. "/net" .. iter .. ".t7")
    net:evaluate()
  
    -- open output file
    local IoUFile = assert(io.open(evalParams.outputFolder .. "/evaluation_" .. splitStr .. objStr .. "_" .. metric .. "_" .. iter .. ".txt", "w"))
    IoUFile:write("modelName")
    for t=1,#thresholds do
      IoUFile:write(", " .. thresholds[t])
    end
    IoUFile:write("\n")
    
    local numEvalFiles = 0
    if splitStr == "val" then
      numEvalFiles = #hierarchical3DShapeNetDataSet.valDataFiles
    else
      numEvalFiles = #hierarchical3DShapeNetDataSet.testDataFiles
    end

        
    for j=1,numEvalFiles do
      if ((j-1)%evalParams.subsampling == 0) then
        local obs, blockIndices, blocks, modelName
	
	if splitStr == "val" then
	  obs, blockIndices, blocks, modelName = hierarchical3DShapeNetDataSet:getNextValExample(false, false)
	else
	  obs, blockIndices, blocks, modelName = hierarchical3DShapeNetDataSet:getNextTestExample(false, false)
	end
	
	class = split(modelName, "/")[1]
       
        IoUFile:write(modelName .. " ")
            
	if networkParameters.useColor then
	  input = obs:view(1,3,128,128):cuda()
	else
	  input = obs:view(1,1,128,128):cuda()
	end
        
        for l=1,numLevels do
          blocks[l] = blocks[l]:cuda()
        end
        
        
        print("Iteration " .. iter .. " running model " .. j .. " : " .. modelName)
            

        -- run network
        local result = net:forward(input)
        numEvaluatedElements[class] = numEvaluatedElements[class] + 1
        
            
        -- compile the ground truth
        groundTruths = {}
        for l=1,numLevels do
          local bISize = blockIndices[l]:size()
          groundTruths[l] = torch.Tensor(1,16*bISize[1],16*bISize[2],16*bISize[3]):cuda()

          for bx=1,bISize[1] do
            for by=1,bISize[2] do
              for bz=1,bISize[3] do
                local bI = blockIndices[l][bx][by][bz]
                groundTruths[l][1][{{(bx-1)*16+1,bx*16},{(by-1)*16+1,by*16},{(bz-1)*16+1,bz*16}}]:copy(blocks[l][bI],1,2)
              end
            end
          end
        end
        
                
        -- upsample the result to GT resolution
        
        local GTRes = groundTruths[numLevels][1]:size()[1]
        
        local upSampled
        
        if (GTRes > 32) then
          local upSampled1 = image.scale(result:double():view(32,32,32),256,256)
          local upSampled1T = upSampled1:transpose(1,3)
          local upSampled2T = image.scale(upSampled1T,256,256)
          upSampled = upSampled2T:transpose(3,1)
          upSampled = upSampled:cuda():view(1,1,256,256,256)
        else
          upSampled = result
        end
	  
        -- compute Error Measures
        
        local labelBoundary
        local labelEDT
        
        if metric == "CD" then
          labelBoundary = groundTruths[numLevels]:clone():view(GTRes,GTRes,GTRes):double()
          hsp.boundary(labelBoundary)
          
          labelEDT = groundTruths[numLevels]:clone():view(GTRes,GTRes,GTRes):double()
          hsp.boundaryEDT(labelEDT)
        end
        
        for t=1,#thresholds do 
          local threshold = thresholds[t]
        
          local binarizedResult = upSampled:clone()
          binarizedResult[torch.le(binarizedResult, threshold)] = 0
          binarizedResult[torch.ge(binarizedResult, threshold)] = 1
	  
	  if (metric == "CD") then
	    if torch.max(binarizedResult) < 1 then
	      local maxVal = torch.max(upSampled)
	      binarizedResult[torch.ge(upSampled,maxVal-1e-5)] = 1
	    end
	  end
        
          -- compute IoU
          local measure
          if metric == "IoU" then
          
            local union = torch.cmax(binarizedResult, groundTruths[numLevels])
            local intersection = torch.cmin(binarizedResult, groundTruths[numLevels])
            local IoU = torch.sum(intersection)/torch.sum(union)
            measure = IoU
            
          elseif metric == "CD" then
            
            local outputVolBoundary = binarizedResult:clone():view(GTRes,GTRes,GTRes):double()
            hsp.boundary(outputVolBoundary)

            local outputEDT = binarizedResult:clone():view(GTRes,GTRes,GTRes):double()
            hsp.boundaryEDT(outputEDT)
            
            local numOutputVolBoundary = torch.sum(outputVolBoundary)
            local numLabelBoundary = torch.sum(labelBoundary)
            
            if (numOutputVolBoundary > 0 and numLabelBoundary > 0) then
                local cd1 = torch.sum(torch.cmul(outputVolBoundary, labelEDT))/(GTRes*numOutputVolBoundary)
                local cd2 = torch.sum(torch.cmul(labelBoundary, outputEDT))/(GTRes*numLabelBoundary)
                --print("cd1 = " .. cd1 .. "  cd2 = " .. cd2)
                measure = (cd1 + cd2)/2
            else
              measure = math.sqrt(3)/2.0
            end
            
          end
        
        
          IoUFile:write(measure .. " ")        
          totalIoU[class][t] = totalIoU[class][t] + measure

        end
      
          IoUFile:write("\n")
          IoUFile:flush()
        
        if (evalParams.saveObj == true) then
          
          hsp.saveMeshAsObj(upSampled[{1,1,{},{},{}}]:double(),evalParams.marchingCubesThreshold, evalParams.outputFolder .. "/objs_" .. splitStr .. "/" .. modelName .. ".obj")
          if networkParameters.useColor then
            image.save(evalParams.outputFolder .. "/objs_" .. splitStr .."/" .. modelName .. "_input.png", obs)
          else
            obs = obs + 0.866
            obs = obs:div(2*0.866)
            image.save(evalParams.outputFolder .. "/objs_" .. splitStr .."/" .. modelName .. "_input.png", obs)
          end
        end
      else
	if splitStr == "val" then
	  hierarchical3DShapeNetDataSet:skipNextValExample()
	else
	  hierarchical3DShapeNetDataSet:skipNextTestExample()
	end
      end
    end
  else
    
    --loading the data from the valDataFiles, computing statistics
    local IoUFile = assert(io.open(evalParams.outputFolder .. "/evaluation_" .. splitStr .. "_" .. metric .. "_" .. iter .. ".txt", "r"))
    titleLine = true
    
    for line in IoUFile:lines() do
      if titleLine then
        titleLine = false
      else
        -- reading data
        elements = split(line, " ")
        class = split(elements[1], "/")[1]
        for t=1,#thresholds do
          totalIoU[class][t] = totalIoU[class][t] + tonumber(elements[t+1])
        end
        numEvaluatedElements[class] = numEvaluatedElements[class] + 1
      end
    end
    
  end
  
  local IoUAvgs = {}
  for t=1,#thresholds do
    IoUAvgs[t] = 0
  end
  
  for c=1,#evalParams.classes do
  
    AvGIoUFiles[evalParams.classes[c]]:write(iter) 
    
    for t=1,#thresholds do
      local classIoU = totalIoU[evalParams.classes[c]][t] / numEvaluatedElements[evalParams.classes[c]]
      AvGIoUFiles[evalParams.classes[c]]:write(" " .. classIoU)
      IoUAvgs[t] = IoUAvgs[t] + classIoU
    end
    AvGIoUFiles[evalParams.classes[c]]:write("\n")
    AvGIoUFiles[evalParams.classes[c]]:flush()
  end
  
  AvGIoUAvgFile:write(iter)
  for t=1,#thresholds do
    local IoUAvg = IoUAvgs[t]/#evalParams.classes
    AvGIoUAvgFile:write(" " .. IoUAvg)
    
    if metric == "IoU" then
      if (IoUAvg > peakPerformance) then
        peakPerformance = IoUAvg
        peakPerformanceIter = iter
        peakPerformanceThresh = thresholds[t]
      end
    elseif metric == "CD" then
      if (IoUAvg < peakPerformance) then
        peakPerformance = IoUAvg
        peakPerformanceIter = iter
        peakPerformanceThresh = thresholds[t]
      end
    end
  end
  AvGIoUAvgFile:write("\n")
  AvGIoUAvgFile:flush()
end

print("peakPerformance = " .. peakPerformance)
print("peakPerformanceIter = " .. peakPerformanceIter)
print("peakPerformanceThresh = " .. peakPerformanceThresh)