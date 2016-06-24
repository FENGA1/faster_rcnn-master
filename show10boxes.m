function show10boxes(im, color_conf, Boxes2save, Label)
% Draw bounding boxes on top of an image.
%   showboxes(im, boxes)
%
% -------------------------------------------------------

fix_width = 800;
if isa(im, 'gpuArray')
    im = gather(im);
end
imsz = size(im);
scale = fix_width / imsz(2);
im = imresize(im, scale);

if ~exist('color_conf', 'var')
    color_conf = 'default';
end

image(im); 
axis image;
axis off;
set(gcf, 'Color', 'white');

if size(Boxes2save,1) > 0
    switch color_conf
        case 'default'
            colors_candidate = colormap('hsv');
            colors_candidate = colors_candidate(1:(floor(size(colors_candidate, 1)/size(Boxes2save,1))):end, :);
            colors_candidate = mat2cell(colors_candidate, ones(size(colors_candidate, 1), 1))';
            colors = cell(size(Boxes2save));
            colors(Boxes2save) = colors_candidate(1:sum(Boxes2save));
        case 'voc'
            colors_candidate = colormap('hsv');
            colors_candidate = colors_candidate(1:(floor(size(colors_candidate, 1)/size(Boxes2save,1))):end, :);
            colors_candidate = mat2cell(colors_candidate, ones(size(colors_candidate, 1), 1))';
            colors = colors_candidate;
        case 'red2white'
            channels = size(Boxes2save,1);
            colors_candidate = [ones(size([1:-1/channels:0]')) [0:1/channels:1]' [0:1/channels:1]'];
            colors_candidate = mat2cell(colors_candidate, ones(size(colors_candidate, 1), 1))';
            colors = colors_candidate;
    end
            

    for i = size(Boxes2save,1):-1:1
        box = Boxes2save(i,:);
        linewidth = 2;
        rectangle('Position', RectLTRB2LTWH(box)*scale, 'LineWidth', linewidth, 'EdgeColor', colors{i});
    end
% label text
    textpos = double(RectLTRB2LTWH(Boxes2save(1,:))*scale);
    text(textpos(1),textpos(2),Label{i});hold on;

end
end

function [ rectsLTWH ] = RectLTRB2LTWH( rectsLTRB )
%rects (l, t, r, b) to (l, t, w, h)

rectsLTWH = [rectsLTRB(:, 1), rectsLTRB(:, 2), rectsLTRB(:, 3)-rectsLTRB(:,1)+1, rectsLTRB(:,4)-rectsLTRB(2)+1];
end

