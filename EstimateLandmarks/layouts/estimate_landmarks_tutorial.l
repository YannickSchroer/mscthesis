[
  [
    "conv2D", 
    {
      "activation": "relu", 
      "init": "uniform", 
      "input_shape": [
        3, 
        48, 
        64
      ], 
      "nb_col": 3, 
      "nb_filter": 32, 
      "nb_row": 3 
    }
  ], 
  [
    "maxpool2D", 
    {
      "pool_size": [
        2, 
        2
      ]
    }
  ], 
  [
    "conv2D", 
    {
      "activation": "relu", 
      "init": "uniform", 
      "nb_col": 2, 
      "nb_filter": 64, 
      "nb_row": 2
    }
  ], 
  [
    "maxpool2D", 
    {
      "pool_size": [
        2, 
        2
      ]
    }
  ], 
  [
    "conv2D", 
    {
      "activation": "relu", 
      "init": "uniform", 
      "nb_col": 2, 
      "nb_filter": 128, 
      "nb_row": 2
    }
  ], 
  [
    "maxpool2D", 
    {
      "pool_size": [
        2, 
        2
      ]
    }
  ], 
  [
    "flatten", 
    {}
  ], 
  [
    "dense", 
    {
      "activation": "relu", 
      "init": "uniform", 
      "output_dim": 500
    }
  ],
  [
  	"dense", 
    {
      "activation": "relu", 
      "init": "uniform", 
      "output_dim": 500
    }
  ],
  [
    "dense", 
    {
      "activation": "linear", 
      "init": "uniform", 
      "output_dim": 4
    }
  ]
]
