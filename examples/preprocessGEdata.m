% script to preprocess stack-of-spirals data acquired on a GE scanner.
% Author: Naveen Murthy (nnmurthy@umich.edu)

%% Setup

% Path to TOPPE (to be modified by user)
addpath('/n/badwater/z/nnmurthy/projects/toppe/')

% Path to BART (to be modified by user)
setenv('TOOLBOX_PATH', '/n/badwater/z/nnmurthy/projects/bart-0.8.00'); % or define this as an environment variable externally
addpath(strcat(getenv('TOOLBOX_PATH'), '/matlab'));

% Path to mirt (to be modified by user)
addpath('/n/badwater/z/nnmurthy/projects/mirt/')
setup;

% Path to other source code
addpath('../GE/')
addpath('../helpers/')

% Paths to data (to be modified by user)
data_dir = '/n/ir71/d2/nnmurthy/data/20221013_UM3TUHP_3dspiral/';
sosp_file = [data_dir 'P,sosp.7'];
b0_file = [data_dir 'P,b0.7'];
readout_b0_file = [data_dir 'readout_b0.mod'];
readout_sosp_file = [data_dir 'readout_sosp.mod'];
output_dir = [data_dir 'out/']; % todo: create directory if needed

if not(isfolder(output_dir))
	warning(sprintf("Output directory does not exist. Creating a folder at %s", output_dir))
	mkdir(output_dir)
end

% imaging setup
res = 0.24;       % resolution in cm
N = [92 92 42];   % image matrix size
fov = N*res;     % FOV in cm
nrot = 3; % number of spiral arms per kz-encode

% samples to discard at the beginning and end of spiral
istart = 114;
istop = 3683;
dwell_time = 4e-3; % sampling time in ms

% flags
estimate_smaps = false;
estimate_b0maps = true;
process_spiral_data = true;

%% Spiral kspace data and trajectories
if process_spiral_data

	% retrospective undersampling if required
	nz = N(3);
	z_undersample_list = 1:nz;
	spiral_undersample_list = [repmat(1:nrot, 1, floor(nz / nrot)), 1:mod(nz, nrot)]; % 3-fold spiral undersampling
	
	% kspace data
	frame_idxs = [4];
	% kdata = readframes(sosp_file, frame_idxs, 'istart', istart, 'istop', istop, 'nrot', nrot,...
	% 	'outfile', [output_dir 'kdata.h5'], 'z_undersample_list', z_undersample_list, 'spiral_undersample_list', spiral_undersample_list);
	kdata = readframes(sosp_file, frame_idxs, 'istart', istart, 'istop', istop, 'nrot', nrot,...
		'outfile', [output_dir 'kdata.h5']);

	% kspace trajectories
	ktraj = getspiraltrajs(readout_sosp_file, fov, N, 'istart', istart, 'istop', istop, ...
		'dwell_time', dwell_time, 'outfile', [output_dir 'ktraj.h5']);
end

%% sensitivity map estimation
if estimate_smaps

	% reconstruct coil images using the first echo from the B0 mapping data. 
	% size(ims) is [nx ny nz ncoils]
	[ims, imsos, d] = toppe.utils.recon3dft(b0_file, ...
	    'readoutFile', readout_b0_file, ...
	    'echo', 1);
	d = d(2:4:end,:,:,:);

	% compute and save sensitivity maps
	smaps = getsensemaps(d, 'outfile', [output_dir 'smaps.h5']);
end

%% field map estimation
if estimate_b0maps

	% Look for sensitivity maps (either as a variable or in the output h5 file.)
	if ~isvar('smaps')
		if isfile([output_dir 'smaps.h5'])
			smaps_r = h5read([output_dir 'smaps.h5'], '/smaps_r');
			smaps_i = h5read([output_dir 'smaps.h5'], '/smaps_i');
			smaps = smaps_r + j*smaps_i;
		else
			error('Sensitivity maps not found. They need to be computed again. Please run the script again with estimate_smaps = true.')
		end
	end

	% Get images at 2 echo times.
	im_te1 = toppe.utils.recon3dft(b0_file, ...
        'echo', 1, ...
        'readoutFile', readout_b0_file, ...
        'flipFid', false); % [nx, ny, nz, ncoil] complex-valued

    im_te2 = toppe.utils.recon3dft(b0_file, ...
	    'echo', 3, ...
	    'readoutFile', readout_b0_file, ...
	    'flipFid', false); % [nx, ny, nz, ncoil] complex-valued

    echotimes = [0 1000/440];  % TE delays (ms)

    % compute fieldmaps
    fieldmap_reg_params = {'l2b', -1};
    b0maps = getb0maps(im_te1, im_te2, echotimes, permute(smaps, [2,3,4,1]), 'fieldmap_reg_params', fieldmap_reg_params,...
     'outfile', [output_dir 'b0maps.h5']);
end