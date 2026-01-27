def Sf_proper(k, pos, batch_size = None):
    #k should be (2,) or (n_ks,2)
    if(len(k.shape) == 1):
        k = np.array([k])
    #pos should be (n_points,2)
    N = pos.shape[0]
    pos_diff = pos[:, None, :] - pos[None, :, :]    #(n_points, n_points, 2)      
    
    if batch_size == None:
        phase = pos_diff@k.T                  #product over exis with length 2. dim is (n_points, n_points, n_ks)    #takes too much space :(
        rho_k = np.exp(-1j * phase)
        return rho_k.sum(axis = (-3,-2)) / N    #mean over points. output is (n_ks,)
    
    
    n_ks = k.shape[0]
    result = np.zeros(n_ks, dtype=complex)
    for start in range(0, n_ks, batch_size):      #takes too much time
        print("batch", start/batch_size , "of", n_ks/batch_size)
        end = min(start + batch_size, n_ks)
        k_batch = k[start:end, :]  # shape (batch, 2)
        
        phase = pos_diff @ k_batch.T
        result[start:end] = np.exp(-1j * phase).sum(axis=(-3, -2))
    
    return result / N
    
    

def get_cartesian_k_grid(n_kx = 200, n_ky = 200, k_max = 2.0):
    kx = np.linspace(-k_max, k_max, n_kx)
    ky = np.linspace(-k_max, k_max, n_ky)

    kx_grid, ky_grid = np.meshgrid(kx, ky)

    # shape (n_kx*n_ky, 2)
    return np.stack([kx_grid.ravel(), ky_grid.ravel()], axis=1) , (n_kx, n_ky, kx, ky)

def display_cartesian_k_grid(S_flat, context):
    n_kx, n_ky, kx, ky = context
    S_2d = S_flat.reshape(n_ky, n_kx)

    plt.figure(figsize=(6,5))
    plt.imshow(np.log(S_2d), 
               extent=[kx.min(), kx.max(), ky.min(), ky.max()],
               origin='lower',
               cmap='viridis')
    plt.colorbar(label='S(k)')
    plt.xlabel("kx")
    plt.ylabel("ky")
    plt.title("Structure Factor S(kx, ky)")
    plt.show()
    
    return S_2d
