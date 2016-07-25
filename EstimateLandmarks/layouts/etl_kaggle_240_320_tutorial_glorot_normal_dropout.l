[
  [
    "conv2D", 
    {
      "activation": "relu", 
      "init": "glorot_normal", 
      "input_shape": [
        3, 
        96, 
        128
      ],
      "nb_filter": 32, 
      "nb_col": 3, 
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
	"dropout",
	{
	  "p": 0.1
	}
  ],
  [
    "conv2D", 
    {
      "activation": "relu", 
      "init": "glorot_normal", 
      "nb_filter": 64,
      "nb_col": 2,  
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
	"dropout",
	{
	  "p": 0.2
	}
  ],
  [
    "conv2D", 
    {
      "activation": "relu", 
      "init": "glorot_normal",
      "nb_filter": 128, 
      "nb_col": 2, 
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
	"dropout",
	{
	  "p": 0.3
	}
  ],
  [
    "flatten", 
    {}
  ], 
  [
    "dense", 
    {
      "activation": "sigmoid", 
      "init": "glorot_normal", 
      "output_dim": 500
    }
  ],
  [
  	"dense", 
    {
      "activation": "sigmoid", 
      "init": "glorot_normal", 
      "output_dim": 500
    }
  ],
  [
	"dropout",
	{
	  "p": 0.5
	}
  ],
  [
    "dense", 
    {
      "activation": "linear", 
      "init": "glorot_normal", 
      "output_dim": 30
    }
  ]
]
