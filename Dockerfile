# docker build -t pest .
# docker run -it -v pest:/usr/pest/data --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --name pest pest
# docker run -i -t -p 8888:8888 continuumio/miniconda3 /bin/bash -c "/opt/conda/bin/conda install jupyter -y --quiet && mkdir /opt/notebooks && /opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser"

FROM continuumio/miniconda3

LABEL maintainer="Ben Evans <ben.d.evans@gmail.com>"

# Set the ENTRYPOINT to use bash (this is also where you’d set SHELL,
# if your version of docker supports this)
ENTRYPOINT [ "/bin/bash", "-c" ]

# Use the environment.yml to create the conda environment.
# https://fmgdata.kinja.com/using-docker-with-conda-environments-1790901398
COPY environment.yml /tmp/environment.yml
RUN [ "conda", "update", "conda", "-y" ]
RUN [ "conda", "update", "--all", "-y" ]

WORKDIR /tmp
# RUN [ "conda", "env", "create" ]
# RUN conda update -n base conda
RUN conda env update -n root -f /tmp/environment.yml
# RUN ["conda", "env", "update", "-n" "root", "-f" "/tmp/environment.yml" ]

# Use bash to source our new environment for setting up private dependencies
# Note that /bin/bash is called in exec mode directly
WORKDIR /usr/pest
#RUN [ "/bin/bash", "-c", "source activate pest && python setup.py develop" ]

# matplotlib config
RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

COPY data /usr/pest/data/
# COPY *.py /usr/pest/pest/
# VOLUME /usr/pest/results

ENV PYTHONPATH "${PYTONPATH}:/usr/pest"
# We set ENTRYPOINT, so while we still use exec mode, we don’t
# explicitly call /bin/bash
# CMD [ "source activate pest && exec python pest/evolution.py" ]
# CMD [ "exec python pest/evolution.py" ]
CMD [ "exec python pest" ]
