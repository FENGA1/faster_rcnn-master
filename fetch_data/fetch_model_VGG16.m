
cur_dir = pwd;
cd(fileparts(mfilename('fullpath')));

try
    fprintf('Downloading model_VGG16...\n');
    urlwrite('https://www.dropbox.com/s/z5rrji25uskha73/model_VGG16.zip?dl=0', 'model_VGG16.zip');
    %onedrive.live.com/download?resid=4006CBB8476FF777!17319&authkey=!APSc546R6TbCCl4&ithint=file%2czip', ...
        

    fprintf('Unzipping...\n');
    unzip('model_VGG16.zip', '..');

    fprintf('Done.\n');
    delete('model_VGG16.zip');
catch
    fprintf('Error in downloading, please try links in README.md https://github.com/ShaoqingRen/faster_rcnn');
end

cd(cur_dir);
