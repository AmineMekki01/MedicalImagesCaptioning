import React, { useState } from 'react';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemText from '@mui/material/ListItemText';
import Divider from '@mui/material/Divider';
import Toolbar from '@mui/material/Toolbar';
import AppBar from '@mui/material/AppBar';
import CssBaseline from '@mui/material/CssBaseline';
import IconButton from '@mui/material/IconButton';
import MenuIcon from '@mui/icons-material/Menu';
import Drawer from '@mui/material/Drawer';
import Button from '@mui/material/Button';

import { styled } from '@mui/material/styles';

import DragAndDropImage from './dragAndDrop';

import MediaCard from './imagecard';

import SwipeableTextMobileStepper from './xrayCarousel';

const drawerWidth = 240;
const navItems = ['Home', 'About Us'];


const mediaCards = [
    <MediaCard
        src={process.env.PUBLIC_URL + '/images/image1.png'}
        desc="A 3-year-old child with visual difficulties. Axial FLAIR image show a supra-sellar lesion extending to the temporal lobes along the optic tracts (arrows) with moderate mass effect, compatible with optic glioma. FLAIR hyperintensity is also noted in the left mesencephalon from additional tumoral involvement"
        title="X-ray chest"
    /> ,
    <MediaCard
    src={process.env.PUBLIC_URL + '/images/image1.png'}
    desc="A 3-year-old child with visual difficulties. Axial FLAIR image show a supra-sellar lesion extending to the temporal lobes along the optic tracts (arrows) with moderate mass effect, compatible with optic glioma. FLAIR hyperintensity is also noted in the left mesencephalon from additional tumoral involvement"
    title="X-ray chest"
    /> ,
    <MediaCard
    src={process.env.PUBLIC_URL + '/images/image1.png'}
    desc="A 3-year-old child with visual difficulties. Axial FLAIR image show a supra-sellar lesion extending to the temporal lobes along the optic tracts (arrows) with moderate mass effect, compatible with optic glioma. FLAIR hyperintensity is also noted in the left mesencephalon from additional tumoral involvement"
    title="X-ray chest"
    /> 
]

const CustomList = styled('ul')({
    listStyle: 'none',
    paddingLeft: 0,
});

const CustomBullet = styled('span')({
    display: 'inline-block',
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    backgroundColor: '#1976d2',
    marginRight: '8px',
});

function DrawerAppBar(props) {
    const { window } = props;
    const [mobileOpen, setMobileOpen] = useState(false);
    const [selectedItem, setSelectedItem] = useState('Home');

    const handleDrawerToggle = () => {
        setMobileOpen((prevState) => !prevState);
    };

    const handleItemClick = (item) => {
        setSelectedItem(item);
        setMobileOpen(false); 
    };

    const getContent = () => {
        if (selectedItem === 'About Us') {
            return (
                <div>

                    <Box sx={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'flex-start' }}>
                        <div>
                            <Typography variant="body1" paragraph sx={{ marginBottom: '20px', lineHeight: '1.6', marginTop: '20px' }}>
                                As Machine Learning Engineers, we understand the challenge of labeling photos.<br />
                                <b>Imagine Radiologists having to  read medical images and write interpretations.</b>
                                Typically, this process can take 5-20 minutes just to review a report. <br /> A doctor might
                                have to assess 40-100 patient images. <br /><br />
                                <b>An automated method to caption medical images could save valuable time for
                                    doctors.</b>
                            </Typography>
                            <Typography variant="h6" sx={{ marginBottom: '10px' }}>
                                Who are we?
                            </Typography>
                            <List sx={{ maxWidth: '300px', marginBottom: '20px' }}>
                                <ListItem>
                                    <CustomBullet />
                                    <ListItemText primary="Amine MEKKI" />
                                </ListItem>
                                <ListItem>
                                    <CustomBullet />
                                    <ListItemText primary="Mohamed EL YOUSFI ALAOUI" />
                                </ListItem>
                                <ListItem>
                                    <CustomBullet />
                                    <ListItemText primary="Kaiyuan GONG" />
                                </ListItem>
                                <ListItem>
                                    <CustomBullet />
                                    <ListItemText primary="Ahmed SIDKI" />
                                </ListItem>
                            </List>
                        </div>
                        <SwipeableTextMobileStepper mediaCards={mediaCards} />

                    </Box>


                </div>
            );
        } else {
            return (
                <Typography>
                    Welcome to your medic
                    <DragAndDropImage />
                </Typography>

            );
        }
    };

    const drawer = (
        <Box onClick={handleDrawerToggle} sx={{ textAlign: 'center' }}>
            <Typography variant="h6" sx={{ my: 2 }}>
                <img src={process.env.PUBLIC_URL + '/images/logo.png'} alt="Logo" style={{ width: '70px', height: '70px', marginRight: '10px' }} />
            </Typography>
            <Divider />
            <List>
                {navItems.map((item) => (
                    <ListItem key={item} disablePadding>
                        <ListItemButton
                            sx={{ textAlign: 'center' }}
                            selected={selectedItem === item}
                            onClick={() => handleItemClick(item)}
                        >
                            <ListItemText primary={item} />
                        </ListItemButton>
                    </ListItem>
                ))}
            </List>
        </Box>
    );

    const container = window !== undefined ? () => window().document.body : undefined;

    return (
        <Box sx={{ display: 'flex' }}>
            <CssBaseline />
            <AppBar component="nav">
                <Toolbar>
                    <IconButton
                        color="inherit"
                        aria-label="open drawer"
                        edge="start"
                        onClick={handleDrawerToggle}
                        sx={{ mr: 2, display: { sm: 'none' } }}
                    >
                        <MenuIcon />
                    </IconButton>
                    <Typography
                        variant="h6"
                        component="div"
                        sx={{ flexGrow: 1, display: { xs: 'none', sm: 'block' } }}
                    >
                        {/* Your logo image */}
                        <img src={process.env.PUBLIC_URL + '/images/logoWhite.png'} alt="Logo" style={{ width: '70px', height: '70px', marginRight: '10px' }} />
                    </Typography>
                    <Box sx={{ display: { xs: 'none', sm: 'block' } }}>
                        {navItems.map((item) => (
                            <Button
                                key={item}
                                sx={{ color: '#fff' }}
                                onClick={() => handleItemClick(item)}
                                variant={selectedItem === item ? 'contained' : 'text'}
                            >
                                {item}
                            </Button>
                        ))}
                    </Box>
                </Toolbar>
            </AppBar>
            <nav>
                <Drawer
                    container={container}
                    variant="temporary"
                    open={mobileOpen}
                    onClose={handleDrawerToggle}
                    ModalProps={{
                        keepMounted: true, // Better open performance on mobile.
                    }}
                    sx={{
                        display: { xs: 'block', sm: 'none' },
                        '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
                    }}
                >
                    {drawer}
                </Drawer>
            </nav>
            <Box component="main" sx={{ p: 3 }}>
                <Toolbar />
                {/* Display content based on the selected item */}
                {getContent()}
            </Box>
        </Box>
    );
}

export default DrawerAppBar;
