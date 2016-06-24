function ImageNet_detection(fast_rcnn_net, opts, proposal_detection_model, rpn_net,folder_index)
%% script for detecting objects from FBMS dataset by Faster RCNN
% set data path
dataDir = '/BS/databases/imagenet/ILSVRC2015_VIS/Data/VID/val/';
Type = '*.JPEG';
outMat = '/BS/joint-multicut-2/work/ImageNet-Video-Object-Detection/Faster-RCNN/';
folder = dir(dataDir);
folder(1:2)=[];

%% -------------------- WARM UP --------------------
% the first run will be slower; use an empty image to warm up

for j = 1:2 % we warm up 2 times
    im = uint8(ones(375, 500, 3)*128);
    if opts.use_gpu
        im = gpuArray(im);
    end
    [boxes, scores]             = proposal_im_detect(proposal_detection_model.conf_proposal, rpn_net, im);
    aboxes                      = boxes_filter([boxes, scores], opts.per_nms_topN, opts.nms_overlap_thres, opts.after_nms_topN, opts.use_gpu);
    if proposal_detection_model.is_share_feature
        [boxes, scores]             = fast_rcnn_conv_feat_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
            rpn_net.blobs(proposal_detection_model.last_shared_output_blob_name), ...
            aboxes(:, 1:4), opts.after_nms_topN);
    else
        [boxes, scores]             = fast_rcnn_im_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
            aboxes(:, 1:4), opts.after_nms_topN);
    end
end


%% detection for each folder
% use folder_index to decide which folder to process
for img_index=folder_index(1):min(size(folder,1),folder_index(2))
    imgDir = [folder(img_index).name '/'];
    Files=dir([dataDir imgDir Type]);
    LengthFiles = length(Files);
    if(~exist([outMat imgDir],'dir'))
        mkdir([outMat imgDir]) ;
    end
    
    for imgnum=1:LengthFiles
        % setting path
        imgName = Files(imgnum).name;
        filename = [dataDir imgDir imgName];
        % in case of JPEG format, replace JPEG with mat for the saved file
        % names
        matName = imgName; matName(end-4:end)='.mat ';matName(end)=[];
        outbox = [outMat imgDir matName];
        Boxes2save = [];
        Label = cell(0);
        % based on faster rcnn demo
        im = imread(filename);
        
        if opts.use_gpu
            im = gpuArray(im);
        end
        
        % test proposal
        [boxes, scores]             = proposal_im_detect(proposal_detection_model.conf_proposal, rpn_net, im);
        aboxes                      = boxes_filter([boxes, scores], opts.per_nms_topN, opts.nms_overlap_thres, opts.after_nms_topN, opts.use_gpu);
        
        % test detection
        if proposal_detection_model.is_share_feature
            [boxes, scores]             = fast_rcnn_conv_feat_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
                rpn_net.blobs(proposal_detection_model.last_shared_output_blob_name), ...
                aboxes(:, 1:4), opts.after_nms_topN);
        else
            [boxes, scores]             = fast_rcnn_im_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
                aboxes(:, 1:4), opts.after_nms_topN);
        end
   
        % visualize
        classes = proposal_detection_model.classes;
        boxes_cell = cell(length(classes), 1);
        thres = 0.6;
        for i = 1:length(boxes_cell)
            boxes_cell{i} = [boxes(:, (1+(i-1)*4):(i*4)), scores(:, i)];
            tempbox = boxes_cell{i}(boxes_cell{i}(:, 5) >= 0.3, :);
            if ~isempty(tempbox)
                templabel = cell(size(tempbox,1),1);
                templabel(:,1) = {classes{i}};
                Label = [Label; templabel];
            end
            Boxes2save = [Boxes2save;tempbox];
            boxes_cell{i} = boxes_cell{i}(nms(boxes_cell{i}, 0.3), :);

            I = boxes_cell{i}(:, 5) >= thres;
            
            boxes_cell{i} = boxes_cell{i}(I, :);
        end
        % choose 10 boxes with largest score
        [~,B_index] = sort(Boxes2save(:,5),'descend');
        if size(B_index,1)>10
            B_index = B_index(1:10);
        end
        % save the boxes
        Boxes2save = Boxes2save(B_index,:);
        Label = Label(B_index,:);
        save(outbox,'Boxes2save','Label');
       
        
        % figure out
        if mod(imgnum-1,20)==0
            h = figure(imgnum);clf;
            set(h,'Visible','off');
            show10boxes(im, 'red2white', Boxes2save, Label);
            img_save = [outMat imgDir imgName];
            fprintf('saving %s\n', img_save);
            print(h,'-dpng', img_save);
            close(h);
        end
    end
    
    
    


end


end % of the function

%% function boxes_filter
function aboxes = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
    % to speed up nms
    if per_nms_topN > 0
        aboxes = aboxes(1:min(length(aboxes), per_nms_topN), :);
    end
    % do nms
    if nms_overlap_thres > 0 && nms_overlap_thres < 1
        aboxes = aboxes(nms(aboxes, nms_overlap_thres, use_gpu), :);       
    end
    if after_nms_topN > 0
        aboxes = aboxes(1:min(length(aboxes), after_nms_topN), :);
    end
end




