import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Embodied Intelligence',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default, // Placeholder - should be replaced with a relevant image
    description: (
      <>
        Learn how intelligence emerges from the interaction between cognitive processes,
        physical body, and environment. Understand morphological computation and
        sensorimotor integration in humanoid systems.
      </>
    ),
  },
  {
    title: 'Modern Robotics Technologies',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default, // Placeholder - should be replaced with a relevant image
    description: (
      <>
        Master ROS 2 (Humble/Iron), Gazebo simulation, NVIDIA Isaac Sim,
        and Vision-Language-Action systems for humanoid robotics development.
      </>
    ),
  },
  {
    title: 'Complete Learning Path',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default, // Placeholder - should be replaced with a relevant image
    description: (
      <>
        From foundational concepts of Physical AI to advanced autonomous
        humanoid implementation. 9 comprehensive chapters with exercises and examples.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
