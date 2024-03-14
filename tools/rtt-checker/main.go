package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/pion/logging"
	"github.com/pion/turn/v3"
)

func main() {
	host := flag.String("host", "", "STUN/TURN Server address")
	port := flag.Int("port", 3478, "Listening port.")
	creds := flag.String("creds", "", "Creadentials: a pair of username and password in a form of \"user=pass\"")
	realm := flag.String("realm", "example.com", "Realm if needed")
	interval := flag.Duration("interval", 250*time.Millisecond, "Interval between RTT probes (default: 250ms)")
	mqttConfigPath := flag.String("mqtt_conf", "mqtt_conf.yaml", "Path to the YAML configuration file with MQTT settings")
	flag.Parse()

	if len(*host) == 0 {
		log.Fatalf("ERROR: 'host' is required")
	} else if len(*creds) == 0 {
		log.Fatalf("ERROR: 'creds' is required")
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, os.Interrupt)

	go func() {
		<-signalChan
		log.Println("OK: Received termination signal. Cancelling...")
		cancel()
	}()

	conn, err := net.ListenPacket("udp4", "0.0.0.0:0")
	if err != nil {
		log.Panicf("ERROR: Failed to listen: %s", err)
	}
	defer func() {
		if closeErr := conn.Close(); closeErr != nil {
			log.Panicf("ERROR: Failed to close connection: %s", closeErr)
		}
	}()

	address := fmt.Sprintf("%s:%d", *host, *port)
	userpass := strings.SplitN(*creds, "=", 2)

	turnCfg := &turn.ClientConfig{
		STUNServerAddr: address,
		TURNServerAddr: address,
		Conn:           conn,
		Username:       userpass[0],
		Password:       userpass[1],
		Realm:          *realm,
		LoggerFactory:  logging.NewDefaultLoggerFactory(),
	}

	turnClient, err := turn.NewClient(turnCfg)
	if err != nil {
		log.Panicf("ERROR: Failed to create a TURN client: %s", err)
	}
	defer turnClient.Close()

	err = turnClient.Listen()
	if err != nil {
		log.Panicf("ERROR: TURN client has failed to listen: %s", err)
	}

	relayedConn, err := turnClient.Allocate()
	if err != nil {
		log.Panicf("ERROR: Failed to send an allocation request: %s", err)
	}
	defer func() {
		if closeErr := relayedConn.Close(); closeErr != nil {
			log.Panicf("Failed to close connection: %s", closeErr)
		}
	}()
	log.Printf("OK: remote-address=%s", relayedConn.LocalAddr().String())

	mqttClient := MqttClient{}
	if err := mqttClient.Init(*mqttConfigPath); err != nil {
		log.Fatalf("Error initializing MQTT client: %v", err)
	}
	defer mqttClient.Close()

	if err := startTurnRttProbingService(ctx, turnClient, relayedConn, &mqttClient, interval); err != nil {
		log.Panicf("ERROR: Ping service has failed: %s", err)
	}
}

func startTurnRttProbingService(ctx context.Context, turnClient *turn.Client, relayedConn net.PacketConn, mqttClient *MqttClient, interval *time.Duration) error {
	log.Printf("OK: Starting RTT probing service with interval %s", interval.String())

	probeConn, err := net.ListenPacket("udp4", "0.0.0.0:0")
	if err != nil {
		return err
	}
	defer probeConn.Close()

	relayedAddress, err := turnClient.SendBindingRequest()
	if err != nil {
		return fmt.Errorf("failed to send a binding request: %w", err)
	}

	// trigger a permission request from the TURN client to accept the traffic by the TURN server
	if _, err := relayedConn.WriteTo([]byte("Hello"), relayedAddress); err != nil {
		return fmt.Errorf("failed to send a permission request: %w", err)
	}

	log.Printf("OK: Initial steps are done, connection is established, starting the probing routines...")

	// listen for the probes from the TURN server to calculate RTT
	go func() {
		buf := make([]byte, 1500)
		for {
			select {
			case <-ctx.Done():
				return
			default:
				nbytes, _, err := probeConn.ReadFrom(buf)
				if err != nil {
					if ctx.Err() != nil {
						return
					}
					log.Printf("ERROR: Failed to read from the probing socket: %s, breaking...", err)
					break
				}

				if sendingTime, err := time.Parse(time.RFC3339Nano, string(buf[:nbytes])); err == nil {
					rtt := time.Since(sendingTime).Milliseconds()

					mqttPayload := MqttPayload{
						Timestamp: time.Now().Format(time.RFC3339Nano),
						ID:        mqttClient.Cfg.Mqtt.ClientID,
						Msg:       rtt,
					}

					jsonPayload, err := json.Marshal(mqttPayload)
					if err != nil {
						if ctx.Err() != nil {
							return
						}
						log.Printf("ERROR: Failed to encode payload as JSON: %s", err)
						return
					}

					mqttClient.Publish(mqttClient.Cfg.Mqtt.Topics.Rtt, jsonPayload, 0, false)
				}
			}
		}
	}()

	// echo the received on the relayed connection probe back to the TURN client
	go func() {
		buf := make([]byte, 1500)
		for {
			select {
			case <-ctx.Done():
				return
			default:
				nbytes, address, err := relayedConn.ReadFrom(buf)
				if err != nil {
					if ctx.Err() != nil {
						return
					}
					log.Printf("ERROR: Failed to read from the relayed connection: %s, breaking...", err)
					return
				}

				if _, err := relayedConn.WriteTo(buf[:nbytes], address); err != nil {
					if ctx.Err() != nil {
						return
					}
					log.Printf("ERROR: Failed to echo the probe back: %s, breaking...", err)
					return
				}
			}
		}
	}()

	// send probes to the TURN server with a given interval
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			default:
				sendingTime := time.Now().Format(time.RFC3339Nano)
				if _, err := probeConn.WriteTo([]byte(sendingTime), relayedConn.LocalAddr()); err != nil {
					if ctx.Err() != nil {
						return
					}
					log.Printf("ERROR: Failed to send a probe: %s, breaking...", err)
					break
				}

				time.Sleep(*interval)
			}
		}
	}()

	<-ctx.Done()

	return nil
}
