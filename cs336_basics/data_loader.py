# ref: 
# - https://github.com/DhyeyMavani2003/stanford-cs336-assignment1-basics-solution/blob/main/cs336_basics/nn_utils.py
import torch

def get_batch(dataset, batch_size: int, context_length: int, device: str):
    """
    Sample a batch of input sequences and their corresponding targets from the dataset.
    
    Args:
        dataset: 1D numpy array of integer token IDs
        batch_size: Number of sequences in the batch
        context_length: Length of each sequence
        device: PyTorch device string (e.g., 'cpu', 'cuda:0', 'mps')
    
    Returns:
        Tuple of (input_sequences, target_sequences) where both are torch.LongTensor
        of shape (batch_size, context_length)
    """
    import numpy as np
    
    # Calculate the maximum valid starting index
    max_start_idx = len(dataset) - context_length
    
    # Randomly sample starting indices
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    
    # Create input and target sequences
    input_sequences = []
    target_sequences = []
    
    for start_idx in start_indices:
        # Input sequence: [start_idx, start_idx + context_length)
        input_seq = dataset[start_idx:start_idx + context_length]
        # Target sequence: [start_idx + 1, start_idx + context_length + 1)
        target_seq = dataset[start_idx + 1:start_idx + context_length + 1]
        
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)
    
    # Convert to numpy arrays and then to tensors
    input_array = np.array(input_sequences)
    target_array = np.array(target_sequences)
    
    # Convert to PyTorch tensors and move to specified device
    input_tensor = torch.from_numpy(input_array).long().to(device)
    target_tensor = torch.from_numpy(target_array).long().to(device)
    
    return input_tensor, target_tensor