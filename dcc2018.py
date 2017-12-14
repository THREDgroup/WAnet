import WAnet

PRE_PROCESS = False
TRAIN = True
EXAMPLES = True
QUICK = False

# Set variables
latent_dim = 16
example_size = (2, 3)

# Preprocess to extract the data
if PRE_PROCESS:
    WAnet.preprocessing.extract_data(1000)

# Train all the models
if TRAIN:
    # WAnet.training.train_response_autoencoder(100, latent_dim, True, True)
    # WAnet.training.train_geometry_autoencoder(40, latent_dim, True, True)
    # WAnet.training.train_forward_network(25, latent_dim, True, True)
    WAnet.training.train_inverse_network(25, latent_dim, True, True)

# if EXAMPLES:
    # WAnet.showing.plot_examples("geometry_autoencoder", example_size[0], example_size[1], quick=QUICK)
    # WAnet.showing.plot_examples("curve_autoencoder", example_size[0], example_size[1], quick=QUICK)
    # WAnet.showing.plot_examples("forward", example_size[0], example_size[1], quick=QUICK)
    # WAnet.showing.plot_examples("inverse", example_size[0], example_size[1], quick=QUICK)
    # WAnet.showing.plot_BIEM_example()