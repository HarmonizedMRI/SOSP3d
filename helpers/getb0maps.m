function b0maps = getb0maps(im_te1, im_te2, echotimes, smaps, varargin)
%function b0maps = getb0maps(im_te1, im_te2, echotimes, smaps, 'option name', option, ...)
%|
%| Estimate fieldmaps from images acquired at two different echo times.
%|
%| Input:
%|   im_te1 : Complex image of size [nx, ny, nz, ncoil] at echo time 1
%|   im_te2 : Complex image of size [nx, ny, nz, ncoil] at echo time 2
%|	 echotimes : 1D array of the two echotimes (in ms)
%|	 smaps : Array of size [nx, ny, nz, ncoil] containing complex-valued sensitivity maps.
%|
%| Options:
%| 	 'outfile' : Path to a .h5 file if k-space data is required to be saved
%|				to file. (Default: "", which does not save to file)
%|	 'fieldmap_reg_params' : Cell array containing keyword arguments that will be passed into
%|							 the mri_field_map_reg() function in mirt toolbox. (Default: {}).
%|							 See mirt/mri/fieldmap/mri_field_map_reg.m for more details.
%|							 Example: fieldmap_reg_params = {'l2b', -1, 'niter', 100}
%|
%| Output:
%|	 b0maps : Array of size [nx, ny, nz] containing fieldmaps (in Hz).

	arg.outfile = ""; % default is to not save to file
	arg.fieldmap_reg_params = {};

	arg = vararg_pair(arg, varargin); % from mirt toolbox

	% coil combined images
    te1 = sum(im_te1 .* conj(smaps),4); 
    te2 = sum(im_te2 .* conj(smaps),4);
    disp('Computing regularized field maps...')
    tic; [wmap, wconv] = mri_field_map_reg(cat(4,te1,te2), echotimes * 1e-3, arg.fieldmap_reg_params{:}); toc;
    b0maps = wmap/2/pi; % convert to Hz
    disp('Done.')

    % save data to file?
	if arg.outfile ~= ""

		% save to file
		[filepath, filename, fileext] = fileparts(arg.outfile);

		if fileext == ".h5"
			if isfile(arg.outfile)
				delete(arg.outfile)
			end

			% create output file
			h5create(arg.outfile, '/b0maps', size(b0maps))

			% write data to file
			disp('Writing field maps to file...')
			h5write(arg.outfile, '/b0maps', b0maps);
			disp('Done.')
		else
			warning(sprintf('Only h5 files are supported currently. Unable to save field maps to file: %s', arg.outfile))
		end

	end

end