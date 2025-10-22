
install_requirements:
	@pip install -r requirements.txt

run_sonic:
	python SonicTheHedgehog2/main_sonic.py --game SonicTheHedgehog2-Genesis --state ${GENESIS_GAME}  --episode 34 --sonic-helper

run_lunar_lander:
	python LunarLander-project-main/main_lander.py --random --episodes 5 --render

retro_sonic:
	sudo apt-get install python3-opengl
	python -m retro.import SonicTheHedgehog2/Sonic\ The\ Hedgehog\ 2\ \(World\)\ \(Rev\ A\)
