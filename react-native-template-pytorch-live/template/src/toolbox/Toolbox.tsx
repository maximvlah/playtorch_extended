/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import {
  CardStyleInterpolators,
  createStackNavigator,
} from '@react-navigation/stack';
import * as React from 'react';
import {ScrollView, StyleSheet} from 'react-native';
import Icon from 'react-native-vector-icons/Ionicons';

import CameraFrameByFrame from './camera/CameraFrameByFrame';
import CameraTakePicture from './camera/CameraTakePicture';

import CanvasShapes from './canvas/CanvasShapes';
import CanvasStarter from './canvas/CanvasStarter';
import CanvasAnimator from './canvas/CanvasAnimator';
import CanvasTransform from './canvas/CanvasTransform';
import Images from './canvas/Images';
import CanvasArcMatrix from './canvas/CanvasArcMatrix';
import CanvasClosePath from './canvas/CanvasClosePath';
import CanvasDrawImage from './canvas/CanvasDrawImage';
import CanvasLineCap from './canvas/CanvasLineCap';
import CanvasLineJoin from './canvas/CanvasLineJoin';
import CanvasMiterLimit from './canvas/CanvasMiterLimit';
import CanvasMoveTo from './canvas/CanvasMoveTo';
import CanvasText from './canvas/CanvasText';
import CanvasTextAlign from './canvas/CanvasTextAlign';

import UIStarter from './ui/UIStarter';
import UICard from './ui/UICard';

import ImageScale from './apiTest/ImageScale';
import CanvasArc from './apiTest/CanvasArc';
import CanvasClearRect from './apiTest/CanvasClearRect';
import CanvasFillRect from './apiTest/CanvasFillRect';
import CanvasRect from './apiTest/CanvasRect';
import CanvasAnimation from './apiTest/CanvasAnimation';
import CanvasRotate from './apiTest/CanvasRotate';
import CanvasSaveRestore from './apiTest/CanvasSaveRestore';
import CanvasSetTransform from './apiTest/CanvasSetTransform';
import CanvasScale from './apiTest/CanvasScale';
import CanvasTranslate from './apiTest/CanvasTranslate';

import ToolboxContext, {useToolboxContext} from './ToolboxContext';
import ToolboxList from './ToolboxList';

export type Tool = {
  icon?: JSX.Element;
  title: string;
  subtitle: string;
  component?: React.ComponentType<any>;
};

export type ToolSection = {
  title: string;
  data: Tool[];
}[];

const tools: ToolSection = [
  {
    title: 'UI',
    data: [
      {
        icon: <Icon name="image-outline" size={32} color="tomato" />,
        title: 'UI Starter',
        subtitle: 'This template helps you start with building UI quickly',
        component: UIStarter,
      },
      {
        icon: <Icon name="image-outline" size={32} color="tomato" />,
        title: 'UI Card',
        subtitle: 'An example gallery of image cards',
        component: UICard,
      },
    ],
  },

  {
    title: 'Canvas',
    data: [
      {
        icon: <Icon name="color-palette-outline" size={32} color="tomato" />,
        title: 'Canvas Starter',
        subtitle: 'This template helps you start with canvas drawing quickly',
        component: CanvasStarter,
      },
      {
        icon: <Icon name="color-palette-outline" size={32} color="tomato" />,
        title: 'Canvas Shapes',
        subtitle: 'Drawing shapes on a canvas',
        component: CanvasShapes,
      },
      {
        icon: <Icon name="color-palette-outline" size={32} color="tomato" />,
        title: 'Canvas Animator',
        subtitle: 'Animator test',
        component: CanvasAnimator,
      },
      {
        icon: <Icon name="color-palette-outline" size={32} color="tomato" />,
        title: 'Canvas Transform',
        subtitle: 'Affine transforms like scale, rotate, and skew',
        component: CanvasTransform,
      },
      {
        icon: <Icon name="color-palette-outline" size={32} color="tomato" />,
        title: 'Canvas#closePath',
        subtitle: 'Close path on a canvas',
        component: CanvasClosePath,
      },
      {
        icon: <Icon name="color-palette-outline" size={32} color="tomato" />,
        title: 'Canvas Arc Matrix',
        subtitle: 'Drawing an arcs matrix on a canvas',
        component: CanvasArcMatrix,
      },
      {
        icon: <Icon name="color-palette-outline" size={32} color="tomato" />,
        title: 'Canvas#moveTo',
        subtitle: 'Drawing lines on a canvas',
        component: CanvasMoveTo,
      },
      {
        icon: <Icon name="color-palette-outline" size={32} color="tomato" />,
        title: 'Canvas#lineCap',
        subtitle: 'Drawing lines with line cap on a canvas',
        component: CanvasLineCap,
      },
      {
        icon: <Icon name="color-palette-outline" size={32} color="tomato" />,
        title: 'Canvas#lineJoin',
        subtitle: 'Drawing lines with line join on a canvas',
        component: CanvasLineJoin,
      },
      {
        icon: <Icon name="color-palette-outline" size={32} color="tomato" />,
        title: 'Canvas#miterLimit',
        subtitle: 'Drawing lines with miter limit on a canvas',
        component: CanvasMiterLimit,
      },
      {
        icon: <Icon name="color-palette-outline" size={32} color="tomato" />,
        title: 'Canvas#drawImage',
        subtitle: 'Drawing images on a canvas',
        component: CanvasDrawImage,
      },
      {
        icon: <Icon name="color-palette-outline" size={32} color="tomato" />,
        title: 'Canvas Text',
        subtitle: 'Drawing text on canvas',
        component: CanvasText,
      },
      {
        icon: <Icon name="color-palette-outline" size={32} color="tomato" />,
        title: 'Canvas#textAlign',
        subtitle: 'Drawing aligned text on canvas',
        component: CanvasTextAlign,
      },
    ],
  },

  {
    title: 'Camera and Images',
    data: [
      {
        icon: <Icon name="image-outline" size={32} color="tomato" />,
        title: 'Images',
        subtitle: 'Different ways to display images',
        component: Images,
      },
      {
        icon: <Icon name="videocam-outline" size={32} color="tomato" />,
        title: 'Camera Frame By Frame',
        subtitle: 'Frame by frame processing of camera images',
        component: CameraFrameByFrame,
      },
      {
        icon: <Icon name="videocam-outline" size={32} color="tomato" />,
        title: 'Camera Take Picture',
        subtitle: 'Take picture processing',
        component: CameraTakePicture,
      },
    ],
  },

  {
    title: 'API Test',
    data: [
      {
        icon: <Icon name="color-palette-outline" size={32} color="tomato" />,
        title: 'Canvas#rect',
        subtitle: 'Drawing rect on a canvas',
        component: CanvasRect,
      },
      {
        icon: <Icon name="color-palette-outline" size={32} color="tomato" />,
        title: 'Canvas#fillRect',
        subtitle: 'Drawing filled rect on a canvas',
        component: CanvasFillRect,
      },
      {
        icon: <Icon name="color-palette-outline" size={32} color="tomato" />,
        title: 'Canvas#clearRect',
        subtitle: 'Clear rect on a canvas',
        component: CanvasClearRect,
      },
      {
        icon: <Icon name="color-palette-outline" size={32} color="tomato" />,
        title: 'Canvas#arc',
        subtitle: 'Drawing arc/circles on a canvas',
        component: CanvasArc,
      },
      {
        icon: <Icon name="color-palette-outline" size={32} color="tomato" />,
        title: 'Canvas Animation',
        subtitle: 'Animate drawings on canvas',
        component: CanvasAnimation,
      },
      {
        icon: <Icon name="color-palette-outline" size={32} color="tomato" />,
        title: 'Canvas Save and Restore',
        subtitle: 'Save, draw, and restore context',
        component: CanvasSaveRestore,
      },
      {
        icon: <Icon name="color-palette-outline" size={32} color="tomato" />,
        title: 'Canvas Set Transform',
        subtitle: 'Draw with transform',
        component: CanvasSetTransform,
      },
      {
        icon: <Icon name="color-palette-outline" size={32} color="tomato" />,
        title: 'Canvas Scale',
        subtitle: 'Drawing with scale on canvas',
        component: CanvasScale,
      },
      {
        icon: <Icon name="color-palette-outline" size={32} color="tomato" />,
        title: 'Canvas Rotate',
        subtitle: 'Drawing with rotate on canvas',
        component: CanvasRotate,
      },
      {
        icon: <Icon name="color-palette-outline" size={32} color="tomato" />,
        title: 'Canvas Translate',
        subtitle: 'Drawing with translate on canvas',
        component: CanvasTranslate,
      },
      {
        icon: <Icon name="image-outline" size={32} color="tomato" />,
        title: 'Image Scale',
        subtitle: 'Scaling images',
        component: ImageScale,
      },
    ],
  },
];

type ToolboxParamList = {
  API: undefined;
  ToolView: {
    title: string;
  };
};

const Stack = createStackNavigator<ToolboxParamList>();

function ToolView() {
  const {activeTool} = useToolboxContext();
  const Component = activeTool?.component;
  return (
    <ScrollView contentContainerStyle={styles.container}>
      {Component && <Component />}
    </ScrollView>
  );
}

export default function Toolbox() {
  return (
    <ToolboxContext>
      <Stack.Navigator
        screenOptions={{
          cardStyleInterpolator: CardStyleInterpolators.forHorizontalIOS,
        }}>
        <Stack.Screen name="API" options={{headerShown: false}}>
          {props => (
            <ToolboxList
              {...props}
              tools={tools}
              onSelect={tool => {
                props.navigation.navigate('ToolView', {
                  title: tool.title,
                });
              }}
            />
          )}
        </Stack.Screen>
        <Stack.Screen
          name="ToolView"
          component={ToolView}
          options={({route}) => {
            if (route != null) {
              return {title: route.params.title};
            }
            return {};
          }}
        />
      </Stack.Navigator>
    </ToolboxContext>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});
