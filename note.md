Starting ur3:

 - payload mettere peso EE, però non importante ora
        => ON

 - ora il robot sblocca i freni, quindi fa click (non ti spaventare)

 - robot mostra "status stopped"
 - fai una prova con "controlled local"  -> in alto a DX c'è un tasto
 - ora metti "controlled remote", che è il pc

Crea connessione:
 - connetti cavo Ethernet del ur3 al pc
 - crea connessione: manuale, IP stesso robot per i primi 3 campi, il terzo metti a caso es. 192.168.1.130, netmask 255.255.255.0 => APPLY

 - prova di connessione su pc: ping 192.168.1.102 , IP del robot (lo trovi su "about")
        => se risponde è andata bene

Far partire drivers su pc PROVE:
 - ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3 robot_ip:=192.168.1.102
 - ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=192.168.0.100 headless_mode:=true
 - ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=192.168.0.100 headless_mode:=true
 - ros2 launch ur_coppeliasim ur_controllers.launch.py ur_type:=ur5e robot_ip:=192.168.0.100 headless_mode:=true controllers_file:=config/ur_controllers_coppelia.yaml

Far partire drivers su pc EFFETTIVI:
 - ros2 launch ur2-launch ur_compliance_controller.launch.py ur_type:=ur5e robot_ip:=192.168.0.100 headless_mode:=true 
 - ros2 launch ur2-launch ur_compliance_controller.launch.py ur_type:=ur3e robot_ip:=192.168.1.102 headless_mode:=true 

Far partire i programmi miei:
 - ros2 run test1 prova (oppure gli altri)