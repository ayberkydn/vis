FROM ayberkydn/deep-learning

# install language related things
RUN pip install black
RUN pip install antialiased-cnns
RUN pip install pytorch_cifar_models

# add user
ARG USERNAME=user
RUN useradd -ms /bin/bash  $USERNAME
USER $USERNAME

