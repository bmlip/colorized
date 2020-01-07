"""
Utility functions for Probabilistic Programming - 3
Wouter Kouw
Last update: 07-01-2020
"""

function one_hot(index::Number, length_vector::Integer)
    "Map integer to one-hot encoding"
    
    # Initialize encoded vector
    encoded = zeros(length_vector,)
    
    # Set i-th index to 1
    encoded[index] = 1.0
    
    return encoded    
end
