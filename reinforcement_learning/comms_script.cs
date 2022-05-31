// https://github.com/CanYouCatchMe01/CSharp-and-Python-continuous-communication

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net;
using System.Threading;
using System.Net.Sockets;
using System.Text;

public class comms_script : MonoBehaviour
{
    Thread mThread;
    public string connectionIP = "127.0.0.1";
    public int connectionPort = 25001;
    IPAddress localAdd;
    TcpListener listener;
    TcpClient client;
    bool step_env=false; 
    int steps_taken = 0;
    Vector3 received_action = Vector3.zero;
    Rigidbody m_Rigidbody;

    bool running;

    private void Update(){
        if(step_env){
            Time.timeScale = 1;
        }
    }

    private void FixedUpdate()
    {
        //transform.localEulerAngles = received_action;
        m_Rigidbody.angularVelocity = received_action;
        steps_taken += 1;
        

        if (steps_taken == 25)
        {
            steps_taken = 0;
            Time.timeScale = 0;
            step_env=false;
        }
    }

    private void Start()
    {
        ThreadStart ts = new ThreadStart(GetInfo);
        mThread = new Thread(ts);
        mThread.Start();
        Time.timeScale = 0;
        m_Rigidbody = GetComponent<Rigidbody>();
    }

    void GetInfo()
    {
        localAdd = IPAddress.Parse(connectionIP);
        listener = new TcpListener(IPAddress.Any, connectionPort);
        listener.Start();

        client = listener.AcceptTcpClient();

        running = true;
        while (running)
        {
            SendAndReceiveData();
        }
        listener.Stop();
    }

    void SendAndReceiveData()
    {
        NetworkStream nwStream = client.GetStream();
        byte[] buffer = new byte[client.ReceiveBufferSize];

        //---receiving Data from the Host----
        int bytesRead = nwStream.Read(buffer, 0, client.ReceiveBufferSize); //Getting data in Bytes from Python
        string dataReceived = Encoding.UTF8.GetString(buffer, 0, bytesRead); //Converting byte data to string

        if (dataReceived != null)
        {
            //---Using received data---
            //Debug.Log(dataReceived);
            received_action = StringToVector3(dataReceived); //<-- assigning receivedPos value from Python
            step_env=true;
            while (step_env){}

            //---Sending Data to Host----
            byte[] myWriteBuffer = Encoding.ASCII.GetBytes("I'm done"); //Converting string to byte data
            nwStream.Write(myWriteBuffer, 0, myWriteBuffer.Length); //Sending the data in Bytes to Python
        }
    }

    public static Vector3 StringToVector3(string sVector)
    {
        // Remove the parentheses
        if (sVector.StartsWith("(") && sVector.EndsWith(")"))
        {
            sVector = sVector.Substring(1, sVector.Length - 2);
        }

        // split the items
        string[] sArray = sVector.Split(',');

        // store as a Vector3
        Vector3 result = new Vector3(
            float.Parse(sArray[0]),
            float.Parse(sArray[1]),
            float.Parse(sArray[2]));

        return result;
    }
}
