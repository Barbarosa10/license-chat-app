import React, { useContext, useState, useRef, useEffect } from 'react'
import { sentimentAnalysisRoute, sendMessageRoute, updateConversationRoute } from "../utils/ApiRoute";
import { useMessages } from "../context/MessageContext";
import {useConversation} from "../context/ConversationContext";
import {useUser} from "../context/UserContext";
import { useChat } from "../context/ChatContext";
import { useVideoCall } from '../context/VideoCallContext';
import axios from "axios";
import { usePopup } from '../context/PopupContext';


const Input = ({videoRef, socket}) => {
    const [text, setText] = useState("");
    const { currentUser } = useUser();
    const { data } = useChat();
    const { myVideo } = useVideoCall();
    const { addMessage } = useMessages();
    const { updateConversation } = useConversation();
    const { showMessage } = usePopup();
    const canvasRef = useRef();

    useEffect(() => {
        // Initialize the canvasRef.current here
        canvasRef.current = document.createElement('canvas');
    }, []);

    const handleSend = async () => {
        const data_to_send = {
            to: data.user._id,
            from: currentUser._id,
            fromUsername: currentUser.username,
            message: text,
            chatId: data.chatId
        }
        console.log(data_to_send);
        socket.current.emit("send-msg", data_to_send);
        const updatedConversation = {
            "id": data.chatId,
            "_id": data.user._id,
            "username": data.user.username,
            "participants": [data.user.username, currentUser.username],
            "lastMessage": text,
            "timestamp": Date.now()
        };
        updateConversation(updatedConversation);

        const message = await axios.post(sendMessageRoute, {
            conversationId: data.chatId,
            sender: currentUser.username,
            message: text,

        });

        if(myVideo.current && canvasRef.current){
            const imageCapture = new ImageCapture(myVideo.current.getVideoTracks()[0]);
            const captureFrame = async () => {
                try {
                  const bitmap = await imageCapture.grabFrame();
          
                  const frameData = await createImageBitmap(bitmap).then((imgBitmap) => {
                    const canvas = document.createElement('canvas');
                    const context = canvas.getContext('2d');
                    canvas.width = imgBitmap.width;
                    canvas.height = imgBitmap.height;
                    context.drawImage(imgBitmap, 0, 0);
                    return canvas.toDataURL('image/jpeg');
                  });

                  if(frameData != undefined){
                    const response = await axios.post(sentimentAnalysisRoute, {
                        text: text,
                        videoFrame: frameData
                    });
                    const { image_sentiment, text_sentiment } = response.data;
                    showMessage("Video sentiment: " + image_sentiment + ", Message sentiment: " + text_sentiment);

                }
                  console.log(frameData);
                } catch (error) {
                  console.error('Error capturing frame:', error);
                }
              };

            await captureFrame();

        }
        // else{
        //     await axios.post(sentimentAnalysisRoute, {
        //         text: text,
        //         videoFrame: "aa"
        //     });
        // }

        addMessage(message.data);

        const conversation = await axios.post(updateConversationRoute, {
            conversationId: data.chatId,
            lastMessage: text
        })
        setText("");
    }

    return(
        <div className='input'>
            <input type="text"
            placeholder='Type something...'
            onChange={(e) => setText(e.target.value)}
            value={text}
            />
            <div className='send'>
                <button onClick={handleSend}>Send</button>
            </div>
        </div>
    )
}

export default Input