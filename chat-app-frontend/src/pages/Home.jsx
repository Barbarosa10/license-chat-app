import React from "react";
import {motion} from "framer-motion"

import Sidebar from "../components/Sidebar";
import Chat from "../components/Chat";


const Home = () => {
    return(
        <motion.div initial={{x: -100, opacity: 0 }} animate={{x: 0, opacity: 1 }} transition={{ duration: 1}} className="home">
            <div className="container">
                <Sidebar/>
                <Chat/>
            </div>
        </motion.div>
    )
}

export default Home