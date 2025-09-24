"""
Enhanced Batch operations functionality for the Enhanced Bruker NMR Data Reader
"""

import numpy as np
import os
import re
from datetime import datetime


class BatchOperations:
    def __init__(self):
        pass

    def add_spectra(self, matrices_dict, selected_spectra):
        """Add selected 2D spectra together with validation"""
        # Validate compatibility first
        valid, message = self.validate_spectra_compatibility(matrices_dict, selected_spectra)
        if not valid:
            return None, 0, message

        # Filter to only use selected spectra
        filtered_2d = []
        for expno, procnos in matrices_dict.items():
            for procno, data_info in procnos.items():
                if f"{expno}/{procno}" in selected_spectra and data_info['data'].ndim == 2:
                    filtered_2d.append(data_info['data'])

        if len(filtered_2d) < 2:
            return None, 0, "Need at least 2 spectra for addition"

        # Add selected spectra
        result = np.zeros_like(filtered_2d[0], dtype=np.complex128)
        count = 0
        for spectrum in filtered_2d:
            if spectrum.shape == result.shape:
                result += spectrum
                count += 1
        
        return result, count, "Addition completed successfully"

    def subtract_spectra(self, matrices_dict, selected_spectra):
        """Subtract selected 2D spectra (first - others) with validation"""
        # Validate compatibility first
        valid, message = self.validate_spectra_compatibility(matrices_dict, selected_spectra)
        if not valid:
            return None, 0, None, message

        # Filter to only use selected spectra
        filtered_2d = []
        for expno, procnos in matrices_dict.items():
            for procno, data_info in procnos.items():
                if f"{expno}/{procno}" in selected_spectra and data_info['data'].ndim == 2:
                    filtered_2d.append((f"{expno}/{procno}", data_info['data']))

        if len(filtered_2d) < 2:
            return None, 0, None, "Need at least 2 spectra for subtraction"

        # Subtract: first spectrum - all others
        result = np.array(filtered_2d[0][1], dtype=np.complex128)
        count = 1
        first_spectrum_name = filtered_2d[0][0]
        
        for i in range(1, len(filtered_2d)):
            spectrum = filtered_2d[i][1]
            if spectrum.shape == result.shape:
                result -= spectrum
                count += 1
        
        return result, count, first_spectrum_name, "Subtraction completed successfully"

    def multiply_spectra(self, matrices_dict, selected_spectra):
        """Multiply selected 2D spectra together with validation"""
        # Validate compatibility first
        valid, message = self.validate_spectra_compatibility(matrices_dict, selected_spectra)
        if not valid:
            return None, 0, message

        # Filter to only use selected spectra
        filtered_2d = []
        for expno, procnos in matrices_dict.items():
            for procno, data_info in procnos.items():
                if f"{expno}/{procno}" in selected_spectra and data_info['data'].ndim == 2:
                    filtered_2d.append(data_info['data'])

        if len(filtered_2d) < 2:
            return None, 0, "Need at least 2 spectra for multiplication"

        # Multiply selected spectra
        result = np.ones_like(filtered_2d[0], dtype=np.complex128)
        count = 0
        for spectrum in filtered_2d:
            if spectrum.shape == result.shape:
                result *= spectrum
                count += 1
        
        return result, count, "Multiplication completed successfully"

    def scalar_multiply_spectrum(self, spectrum_data, scalar):
        """Multiply spectrum by a scalar value"""
        try:
            scalar_val = float(scalar)
            result = spectrum_data * scalar_val
            return result, f"Multiplied by {scalar_val}"
        except ValueError:
            return None, f"Invalid scalar value: {scalar}"

    def scalar_divide_spectrum(self, spectrum_data, divisor):
        """Divide spectrum by a scalar value"""
        try:
            divisor_val = float(divisor)
            if divisor_val == 0:
                return None, "Cannot divide by zero"
            result = spectrum_data / divisor_val
            return result, f"Divided by {divisor_val}"
        except ValueError:
            return None, f"Invalid divisor value: {divisor}"

    def average_spectra(self, matrices_dict, selected_spectra):
        """Calculate average of selected 2D spectra with validation"""
        result, count, message = self.add_spectra(matrices_dict, selected_spectra)
        if result is not None and count > 0:
            result = result / count
            message = f"Averaged {count} spectra successfully"
        return result, count, message
    
    def nabla_spectrum(self, matrices_dict, spectrum_id):
        """Calculate gradient (nabla) of a single 2D spectrum with validation"""
        expno, procno = spectrum_id.split('/')
        if (expno not in matrices_dict or procno not in matrices_dict[expno]):
            return None, None, f"Spectrum {spectrum_id} not found"
        
        data = matrices_dict[expno][procno]['data']
        if data.ndim != 2:
            return None, None, f"Spectrum {spectrum_id} is not 2D"
        
        # Calculate gradients along both axes
        grad_y, grad_x = np.gradient(data.real)  # Use real part for gradient
        
        return grad_x, grad_y, "Gradient calculated successfully"

    def dot_product_spectra(self, matrices_dict, selected_spectra):
        """Calculate dot product of exactly 2 spectra (returns scalar) with validation"""
        if len(selected_spectra) != 2:
            return None, 0, "Need exactly 2 spectra for dot product"
        
        # Validate compatibility
        valid, message = self.validate_spectra_compatibility(matrices_dict, selected_spectra)
        if not valid:
            return None, 0, message
        
        spectra = []
        for spectrum_id in selected_spectra:
            expno, procno = spectrum_id.split('/')
            if (expno in matrices_dict and procno in matrices_dict[expno]):
                data = matrices_dict[expno][procno]['data']
                spectra.append(data.flatten())  # Flatten for dot product
        
        if len(spectra) != 2 or spectra[0].shape != spectra[1].shape:
            return None, 0, "Shape mismatch for dot product"
        
        # Calculate dot product
        dot_result = np.vdot(spectra[0], spectra[1])  # Complex conjugate dot product
        
        return dot_result, 2, "Dot product calculated successfully"

    def cross_product_spectra(self, matrices_dict, selected_spectra):
        """Calculate elementwise cross product of exactly 2 spectra with validation"""
        if len(selected_spectra) != 2:
            return None, 0, "Need exactly 2 spectra for cross product"
        
        # Validate compatibility
        valid, message = self.validate_spectra_compatibility(matrices_dict, selected_spectra)
        if not valid:
            return None, 0, message
        
        spectra = []
        for spectrum_id in selected_spectra:
            expno, procno = spectrum_id.split('/')
            if (expno in matrices_dict and procno in matrices_dict[expno]):
                spectra.append(matrices_dict[expno][procno]['data'])
        
        if len(spectra) != 2 or spectra[0].shape != spectra[1].shape:
            return None, 0, "Shape mismatch for cross product"
        
        # Elementwise cross product (treating as complex numbers)
        result = spectra[0] * np.conj(spectra[1]) - np.conj(spectra[0]) * spectra[1]
        
        return result, 2, "Cross product calculated successfully"

    def convolve_spectrum_with_kernel(self, matrices_dict, spectrum_id, kernel_data):
        """Convolve a spectrum with an external kernel with validation"""
        expno, procno = spectrum_id.split('/')
        if (expno not in matrices_dict or procno not in matrices_dict[expno]):
            return None, f"Spectrum {spectrum_id} not found"
        
        spectrum_data = matrices_dict[expno][procno]['data']
        
        try:
            from scipy import ndimage
            # Perform convolution
            if spectrum_data.dtype == np.complex128:
                # Handle complex data by convolving real and imaginary parts separately
                real_conv = ndimage.convolve(spectrum_data.real, kernel_data, mode='constant')
                imag_conv = ndimage.convolve(spectrum_data.imag, kernel_data, mode='constant')
                result = real_conv + 1j * imag_conv
            else:
                result = ndimage.convolve(spectrum_data, kernel_data, mode='constant')
            
            return result, "Convolution completed successfully"
        except ImportError:
            # Fallback to numpy if scipy not available
            result = np.convolve(spectrum_data.flatten(), kernel_data.flatten(), mode='same').reshape(spectrum_data.shape)
            return result, "Convolution completed (numpy fallback)"
        except Exception as e:
            return None, f"Convolution failed: {str(e)}"

    def validate_file_save(self, filepath, data):
        """Validate that a file was saved correctly"""
        try:
            if not os.path.exists(filepath):
                return False, f"File not created: {filepath}"
            
            # Try to reload and verify
            loaded_data = np.load(filepath)
            if loaded_data.shape != data.shape:
                return False, f"Shape mismatch in saved file: expected {data.shape}, got {loaded_data.shape}"
            
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                return False, f"Empty file created: {filepath}"
            
            return True, f"File saved successfully: {os.path.basename(filepath)} ({file_size} bytes)"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def get_compatible_spectra(self, matrices_dict, reference_spectrum_id):
        """Get list of spectra compatible with reference spectrum for batch operations"""
        if not reference_spectrum_id:
            return []
        
        ref_expno, ref_procno = reference_spectrum_id.split('/')
        if (ref_expno not in matrices_dict or 
            ref_procno not in matrices_dict[ref_expno]):
            return []
        
        ref_data = matrices_dict[ref_expno][ref_procno]['data']
        ref_shape = ref_data.shape
        ref_ndim = ref_data.ndim
        
        compatible = []
        for expno, procnos in matrices_dict.items():
            for procno, data_info in procnos.items():
                spectrum_id = f"{expno}/{procno}"
                data = data_info['data']
                
                # Check compatibility
                if (data.ndim == ref_ndim and 
                    data.shape == ref_shape and
                    spectrum_id != reference_spectrum_id):
                    compatible.append(spectrum_id)
        
        return compatible

    def validate_spectra_compatibility(self, matrices_dict, selected_spectra):
        """Validate that all selected spectra are compatible for batch operations"""
        if len(selected_spectra) < 1:
            return False, "No spectra selected"
        
        if len(selected_spectra) == 1:
            # Single spectrum operations (like gradient) are always valid if spectrum exists
            spectrum_id = selected_spectra[0]
            expno, procno = spectrum_id.split('/')
            if (expno not in matrices_dict or procno not in matrices_dict[expno]):
                return False, f"Spectrum {spectrum_id} not found"
            return True, "Single spectrum validated"
        
        # Get reference spectrum (first one)
        ref_id = selected_spectra[0]
        ref_expno, ref_procno = ref_id.split('/')
        
        if (ref_expno not in matrices_dict or 
            ref_procno not in matrices_dict[ref_expno]):
            return False, f"Reference spectrum {ref_id} not found"
        
        ref_data = matrices_dict[ref_expno][ref_procno]['data']
        ref_shape = ref_data.shape
        ref_ndim = ref_data.ndim
        
        # Check all other spectra against reference
        for spectrum_id in selected_spectra[1:]:
            expno, procno = spectrum_id.split('/')
            
            if (expno not in matrices_dict or 
                procno not in matrices_dict[expno]):
                return False, f"Spectrum {spectrum_id} not found"
            
            data = matrices_dict[expno][procno]['data']
            
            if data.ndim != ref_ndim:
                return False, f"Dimension mismatch: {spectrum_id} has {data.ndim}D vs reference {ref_ndim}D"
            
            if data.shape != ref_shape:
                return False, f"Shape mismatch: {spectrum_id} has shape {data.shape} vs reference {ref_shape}"
        
        return True, f"All {len(selected_spectra)} spectra are compatible"
    
    def execute_macro_script(self, script_content):
        """Execute enhanced TopSpin-style macro script with scalar operations"""
        import os
        import re
        from datetime import datetime
        
        results = {'processed': 0, 'skipped': [], 'errors': [], 'messages': []}
        variables = {}
        loaded_spectra = {}
        
        # Parse script line by line
        lines = [line.strip() for line in script_content.split('\n') if line.strip() and not line.strip().startswith('#')]
        
        try:
            i = 0
            while i < len(lines):
                line = lines[i]
                
                # Variable substitution
                for var, value in variables.items():
                    line = line.replace(f'${var}', str(value))
                
                # Parse commands
                if line.startswith('SET '):
                    # SET variable value
                    parts = line[4:].split(' ', 1)
                    var_name = parts[0]
                    var_value = parts[1].strip('"')
                    variables[var_name] = var_value
                    
                elif line.startswith('LOOP EXPNO IN '):
                    # LOOP EXPNO IN path
                    input_path = line[14:].strip().strip('"')
                    if input_path.startswith('$'):
                        input_path = variables.get(input_path[1:], input_path)
                    
                    # Find loop end
                    loop_end = self.find_loop_end(lines, i)
                    loop_body = lines[i+1:loop_end]
                    
                    # Execute loop for each EXPNO
                    if os.path.exists(input_path):
                        expno_folders = [d for d in os.listdir(input_path) 
                                    if os.path.isdir(os.path.join(input_path, d)) and d.isdigit()]
                        
                        for expno in expno_folders:
                            variables['EXPNO'] = expno
                            self.execute_loop_body(loop_body, variables, loaded_spectra, results, input_path)
                            
                    i = loop_end  # Skip to after loop
                    
                elif line.startswith('PRINT '):
                    # PRINT message
                    message = line[6:].strip('"')
                    results['messages'].append(message)
                
                elif line.startswith('LOAD '):
                    # LOAD expno/procno AS alias
                    self.execute_load_command(line, variables, loaded_spectra, results)
                
                elif any(op in line for op in [' AS ', 'ADD ', 'SUBTRACT ', 'MULTIPLY ', 'DIVIDE ']):
                    # Operation commands
                    self.execute_operation_command(line, loaded_spectra, results)
                    
                elif line.startswith('SAVE '):
                    # SAVE result TO "path"
                    self.execute_save_command(line, loaded_spectra, variables, results)
                
                i += 1
                
        except Exception as e:
            results['errors'].append(f"Script execution error: {str(e)}")
        
        # Set output path for results display
        results['output_path'] = variables.get('OUTPUT_PATH', 'Unknown')
        
        return results

    def execute_load_command(self, command, variables, loaded_spectra, results):
        """Execute LOAD command outside of loop context"""
        try:
            parts = command[5:].split(' AS ')
            spectrum_path = parts[0].strip()
            alias = parts[1].strip()
            
            # For standalone LOAD commands, assume .npy files in current directory
            if os.path.exists(spectrum_path + '.npy'):
                loaded_spectra[alias] = np.load(spectrum_path + '.npy')
                results['messages'].append(f"Loaded {spectrum_path}.npy as {alias}")
            else:
                results['errors'].append(f"Could not load {spectrum_path}")
        except Exception as e:
            results['errors'].append(f"Load command error: {str(e)}")

    def execute_save_command(self, command, loaded_spectra, variables, results):
        """Execute SAVE command with file validation"""
        try:
            parts = command[5:].split(' TO ')
            alias = parts[0].strip()
            file_path = parts[1].strip().strip('"')
            
            # Variable substitution in path
            for var, value in variables.items():
                file_path = file_path.replace(f'${var}', str(value))
            
            if alias in loaded_spectra:
                # Ensure directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Save and validate
                np.save(file_path, loaded_spectra[alias])
                valid, message = self.validate_file_save(file_path, loaded_spectra[alias])
                
                if valid:
                    results['messages'].append(f"Saved {alias} to {os.path.basename(file_path)}")
                else:
                    results['errors'].append(f"Save validation failed: {message}")
            else:
                results['errors'].append(f"Alias {alias} not found for saving")
                
        except Exception as e:
            results['errors'].append(f"Save command error: {str(e)}")

    def find_loop_end(self, lines, start_index):
        """Find the END LOOP statement"""
        for i in range(start_index + 1, len(lines)):
            if lines[i].strip() == 'END LOOP':
                return i
        return len(lines)

    def execute_loop_body(self, loop_body, variables, loaded_spectra, results, input_path):
        """Execute the body of a loop with enhanced error handling"""
        import os
        
        expno = variables.get('EXPNO', '')
        
        try:
            for line in loop_body:
                # Variable substitution
                for var, value in variables.items():
                    line = line.replace(f'${var}', str(value))
                
                if line.startswith('LOAD '):
                    # LOAD expno/procno AS alias
                    parts = line[5:].split(' AS ')
                    spectrum_path = parts[0].strip()
                    alias = parts[1].strip()
                    
                    # Try to load spectrum (simplified)
                    full_path = os.path.join(input_path, spectrum_path)
                    if os.path.exists(full_path):
                        # This would use your actual data loader
                        npy_file = os.path.join(full_path, 'data.npy')
                        if os.path.exists(npy_file):
                            loaded_spectra[alias] = np.load(npy_file)
                    
                elif any(op in line for op in [' AS ', 'ADD ', 'SUBTRACT ', 'MULTIPLY ', 'DIVIDE ']):
                    # Operation: ADD spec1 spec2 AS result or MULTIPLY result 0.5 AS scaled
                    self.execute_operation_command(line, loaded_spectra, results)
                    
                elif line.startswith('SAVE '):
                    # SAVE result TO "path"
                    self.execute_save_command(line, loaded_spectra, variables, results)
            
            results['processed'] += 1
            
        except Exception as e:
            results['errors'].append(f"EXPNO {expno}: {str(e)}")

    def execute_operation_command(self, command, loaded_spectra, results):
        """Execute enhanced operation command with scalar operations"""
        try:
            if command.startswith('ADD '):
                parts = command[4:].split(' AS ')
                operands = parts[0].strip().split()
                result_alias = parts[1].strip()
                
                if len(operands) >= 2 and all(op in loaded_spectra for op in operands):
                    result = loaded_spectra[operands[0]].copy()
                    for op in operands[1:]:
                        result = result + loaded_spectra[op]
                    loaded_spectra[result_alias] = result
                    results['messages'].append(f"Added {' + '.join(operands)} as {result_alias}")
            
            elif command.startswith('SUBTRACT '):
                parts = command[9:].split(' AS ')
                operands = parts[0].strip().split()
                result_alias = parts[1].strip()
                
                if len(operands) >= 2 and all(op in loaded_spectra for op in operands):
                    result = loaded_spectra[operands[0]].copy()
                    for op in operands[1:]:
                        result = result - loaded_spectra[op]
                    loaded_spectra[result_alias] = result
                    results['messages'].append(f"Subtracted {operands[0]} - {' - '.join(operands[1:])} as {result_alias}")
            
            elif command.startswith('MULTIPLY '):
                parts = command[9:].split(' AS ')
                operands = parts[0].strip().split()
                result_alias = parts[1].strip()
                
                # Check if this is scalar multiplication (spectrum * number)
                if len(operands) == 2:
                    spectrum_name = operands[0]
                    try:
                        scalar = float(operands[1])
                        if spectrum_name in loaded_spectra:
                            result, message = self.scalar_multiply_spectrum(loaded_spectra[spectrum_name], scalar)
                            if result is not None:
                                loaded_spectra[result_alias] = result
                                results['messages'].append(f"Multiplied {spectrum_name} by {scalar} as {result_alias}")
                            else:
                                results['errors'].append(f"Scalar multiplication failed: {message}")
                        else:
                            results['errors'].append(f"Spectrum {spectrum_name} not found for multiplication")
                    except ValueError:
                        # Not a scalar, treat as spectrum multiplication
                        if all(op in loaded_spectra for op in operands):
                            result = loaded_spectra[operands[0]].copy()
                            for op in operands[1:]:
                                result = result * loaded_spectra[op]
                            loaded_spectra[result_alias] = result
                            results['messages'].append(f"Multiplied {' * '.join(operands)} as {result_alias}")
                else:
                    # Multiple spectrum multiplication
                    if all(op in loaded_spectra for op in operands):
                        result = loaded_spectra[operands[0]].copy()
                        for op in operands[1:]:
                            result = result * loaded_spectra[op]
                        loaded_spectra[result_alias] = result
                        results['messages'].append(f"Multiplied {' * '.join(operands)} as {result_alias}")
            
            elif command.startswith('DIVIDE '):
                parts = command[7:].split(' AS ')
                operands = parts[0].strip().split()
                result_alias = parts[1].strip()
                
                # Scalar division (spectrum / number)
                if len(operands) == 2:
                    spectrum_name = operands[0]
                    try:
                        divisor = float(operands[1])
                        if spectrum_name in loaded_spectra:
                            result, message = self.scalar_divide_spectrum(loaded_spectra[spectrum_name], divisor)
                            if result is not None:
                                loaded_spectra[result_alias] = result
                                results['messages'].append(f"Divided {spectrum_name} by {divisor} as {result_alias}")
                            else:
                                results['errors'].append(f"Scalar division failed: {message}")
                        else:
                            results['errors'].append(f"Spectrum {spectrum_name} not found for division")
                    except ValueError:
                        results['errors'].append(f"Invalid divisor: {operands[1]}")
                        
        except Exception as e:
            results['errors'].append(f"Operation command error: {str(e)}")
    
    def get_batch_statistics(self, matrices_dict, selected_spectra):
        """Get statistics for batch operation preview with validation"""
        if not selected_spectra:
            return {'error': 'No spectra selected'}
        
        stats = {
            'count': len(selected_spectra),
            'dimensions': [],
            'shapes': [],
            'dtypes': [],
            'size_mb': 0,
            'missing': [],
            'compatible': True
        }
        
        ref_shape = None
        ref_ndim = None
        
        for spectrum_id in selected_spectra:
            expno, procno = spectrum_id.split('/')
            if (expno in matrices_dict and 
                procno in matrices_dict[expno]):
                
                data = matrices_dict[expno][procno]['data']
                stats['dimensions'].append(data.ndim)
                stats['shapes'].append(data.shape)
                stats['dtypes'].append(str(data.dtype))
                stats['size_mb'] += data.nbytes / (1024 * 1024)
                
                # Check compatibility
                if ref_shape is None:
                    ref_shape = data.shape
                    ref_ndim = data.ndim
                elif data.shape != ref_shape or data.ndim != ref_ndim:
                    stats['compatible'] = False
            else:
                stats['missing'].append(spectrum_id)
        
        return stats
    
    def generate_combined_frames(self, matrices_dict, selected_expnos, save_location):
        """Generate combined frames from PROCNO 231 and 232 with enhanced validation"""
        import os
        from datetime import datetime
        import numpy as np
        
        # Create main folder with current date
        date_str = datetime.now().strftime("%Y%m%d")
        main_folder = os.path.join(save_location, f"NMR_frame_{date_str}")
        frames_463_folder = os.path.join(main_folder, "Frames_463")
        frames_1_folder = os.path.join(main_folder, "Frames_1")
        
        # Create directories
        os.makedirs(frames_463_folder, exist_ok=True)
        os.makedirs(frames_1_folder, exist_ok=True)
        
        results = {'processed': 0, 'skipped': [], 'errors': [], 'saved_files': []}
        
        for expno in selected_expnos:
            try:
                # Check if both PROCNO 231 and 232 exist
                if (expno not in matrices_dict or 
                    '231' not in matrices_dict[expno] or 
                    '232' not in matrices_dict[expno]):
                    results['skipped'].append(f"{expno} - Missing PROCNO 231 or 232")
                    continue
                
                # Load matrices
                m_231 = matrices_dict[expno]['231']['data']
                m_232 = matrices_dict[expno]['232']['data']
                
                # Check shape compatibility
                if m_231.shape != m_232.shape:
                    results['errors'].append(f"{expno} - Shape mismatch between 231 and 232")
                    continue
                
                # Compute combined matrices
                m_add = m_231 + m_232  # For _463
                m_sub = m_231 - m_232  # For _1
                
                # Save results with validation
                add_filename = f"{expno}_463.npy"
                sub_filename = f"{expno}_1.npy"
                
                add_path = os.path.join(frames_463_folder, add_filename)
                sub_path = os.path.join(frames_1_folder, sub_filename)
                
                np.save(add_path, m_add)
                np.save(sub_path, m_sub)
                
                # Validate saves
                add_valid, add_msg = self.validate_file_save(add_path, m_add)
                sub_valid, sub_msg = self.validate_file_save(sub_path, m_sub)
                
                if add_valid and sub_valid:
                    results['saved_files'].extend([add_filename, sub_filename])
                    
                    # Store in matrices_dict for potential reloading
                    if 'frame_results' not in matrices_dict:
                        matrices_dict['frame_results'] = {}
                    
                    matrices_dict['frame_results'][f"{expno}_463"] = {
                        'data': m_add,
                        'dic': matrices_dict[expno]['231']['dic'].copy(),
                        'path': add_path
                    }
                    
                    matrices_dict['frame_results'][f"{expno}_1"] = {
                        'data': m_sub,
                        'dic': matrices_dict[expno]['231']['dic'].copy(),
                        'path': sub_path
                    }
                    
                    results['processed'] += 1
                else:
                    if not add_valid:
                        results['errors'].append(f"{expno}_463 - {add_msg}")
                    if not sub_valid:
                        results['errors'].append(f"{expno}_1 - {sub_msg}")
                
            except Exception as e:
                results['errors'].append(f"{expno} - Error: {str(e)}")
        
        return results, main_folder

    def export_to_npy(self, matrices_dict, selected_spectra, save_location):
        """Export selected spectra to .npy files with validation"""
        import os